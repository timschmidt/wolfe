use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::{collections::HashSet, env};

use arrow_array::types::Float32Type;
use arrow_array::{ArrayRef, FixedSizeListArray, RecordBatch, RecordBatchIterator, StringArray, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use clap::Parser;
use futures::StreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, Table, connect};
use notify::{Event, RecursiveMode, Watcher, recommended_watcher};
use serde::Deserialize;
use serde_json::json;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to a file or directory to embed recursively
    #[arg(short, long, value_name = "PATH", alias = "file", conflicts_with = "search")]
    path: Option<PathBuf>,

    /// Query string to vectorize and search semantically
    #[arg(long, value_name = "TEXT", conflicts_with = "path")]
    search: Option<String>,

    /// Path to the local model directory
    #[arg(long, default_value = "jina-embeddings-v4")]
    model_dir: PathBuf,

    /// Embedding task name
    #[arg(long, default_value = "retrieval")]
    task: String,

    /// Path to the Lance table directory
    #[arg(long, default_value = "wolfe.lance")]
    db: PathBuf,

    /// Path to the Python interpreter
    #[arg(long, default_value = "python3")]
    python: String,

    /// Execution device: auto, cpu, cuda, or mps
    #[arg(long, default_value = "auto")]
    device: String,

    /// Path to the embedding helper script
    #[arg(long, default_value = "scripts/embed.py")]
    script: PathBuf,

    /// Maximum number of search results to return
    #[arg(long, default_value_t = 10)]
    limit: usize,

    /// Return search results as JSON
    #[arg(long)]
    json: bool,

    /// Range of search results to return, formatted as START:END (0-based, end-exclusive)
    #[arg(long, value_name = "START:END")]
    range: Option<String>,

    /// For non-English audio, also run a second Whisper pass forced to English
    #[arg(long)]
    translate: bool,

    /// Max VRAM for Qwen in MB (used with device_map=auto)
    #[arg(long, value_name = "MB")]
    qwen_max_memory: Option<usize>,

    /// Keep only one large PyTorch model in VRAM at a time and force YAMNet onto CPU
    #[arg(long)]
    low_memory: bool,

    /// File or directory names or paths to ignore during ingest and watch
    #[arg(long, value_name = "PATH", action = clap::ArgAction::Append)]
    ignore: Vec<PathBuf>,

    /// File containing newline-separated file or directory names or paths to ignore
    #[arg(long, value_name = "FILE")]
    ignore_file: Option<PathBuf>,

    /// Watch for changes and keep the index up to date
    #[arg(long, requires = "path", conflicts_with = "search")]
    watch: bool,
}

#[derive(Debug, Deserialize)]
struct WorkerRecord {
    record_id: Option<String>,
    path: String,
    file_name: Option<String>,
    extension: Option<String>,
    parent_dir: Option<String>,
    modality: Option<String>,
    chunk: Option<u32>,
    offset: Option<u64>,
    plaintext: Option<String>,
    size_bytes: Option<u64>,
    modified_at: Option<String>,
    embedding: Option<Vec<f32>>,
    reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct QueryRecord {
    status: String,
    embedding: Option<Vec<f32>>,
    reason: Option<String>,
}

#[derive(Debug)]
enum Mode<'a> {
    Ingest(&'a Path),
    Search(&'a str),
}

struct TableTarget {
    db_root: PathBuf,
    table_name: String,
    table_path: PathBuf,
}

struct Summary {
    stored: usize,
    skipped: usize,
    errors: usize,
}

fn parse_range(value: &str) -> Result<(usize, usize), String> {
    let trimmed = value.trim();
    let (start_str, end_str) = trimmed
        .split_once(':')
        .ok_or_else(|| "range must be formatted as START:END".to_string())?;
    if start_str.trim().is_empty() || end_str.trim().is_empty() {
        return Err("range must be formatted as START:END".to_string());
    }
    let start = start_str
        .trim()
        .parse::<usize>()
        .map_err(|_| "range START must be a non-negative integer".to_string())?;
    let end = end_str
        .trim()
        .parse::<usize>()
        .map_err(|_| "range END must be a non-negative integer".to_string())?;
    if end <= start {
        return Err("range END must be greater than START".to_string());
    }
    Ok((start, end))
}

struct WorkerSession {
    child: std::process::Child,
    writer: BufWriter<std::process::ChildStdin>,
    reader: BufReader<std::process::ChildStdout>,
}

struct IgnoreMatcher {
    names: HashSet<String>,
    paths: Vec<PathBuf>,
}

fn resolve_table_target(db_path: &Path) -> TableTarget {
    if db_path.extension().and_then(|ext| ext.to_str()) == Some("lance") {
        let db_root = match db_path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
            _ => PathBuf::from("."),
        };
        let table_name = db_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("wolfe")
            .to_string();
        let table_path = db_root.join(format!("{table_name}.lance"));
        return TableTarget {
            db_root,
            table_name,
            table_path,
        };
    }

    TableTarget {
        db_root: db_path.to_path_buf(),
        table_name: "embeddings".to_string(),
        table_path: db_path.join("embeddings.lance"),
    }
}

fn normalize_ignore_path(path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    Ok(env::current_dir()?.join(path))
}

fn load_ignore_matcher(args: &Args) -> Result<IgnoreMatcher, Box<dyn std::error::Error>> {
    let mut raw_entries = Vec::new();
    raw_entries.extend(args.ignore.iter().cloned());

    if let Some(ignore_file) = &args.ignore_file {
        let content = fs::read_to_string(ignore_file)?;
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            raw_entries.push(PathBuf::from(trimmed));
        }
    }

    let mut names = HashSet::new();
    let mut paths = Vec::new();

    for entry in raw_entries {
        let entry_str = entry.to_string_lossy();
        if !entry_str.contains(std::path::MAIN_SEPARATOR) && !entry.is_absolute() {
            names.insert(entry_str.to_string());
        }
        paths.push(normalize_ignore_path(&entry)?);
    }

    Ok(IgnoreMatcher { names, paths })
}

fn path_matches_ignore_name(path: &Path, matcher: &IgnoreMatcher) -> bool {
    path.components().any(|component| match component {
        Component::Normal(name) => matcher.names.contains(&name.to_string_lossy().to_string()),
        _ => false,
    })
}

fn path_matches_ignore_path(path: &Path, matcher: &IgnoreMatcher) -> bool {
    matcher.paths.iter().any(|ignored| path.starts_with(ignored))
}

fn should_ignore_path(path: &Path, matcher: &IgnoreMatcher) -> bool {
    path_matches_ignore_name(path, matcher) || path_matches_ignore_path(path, matcher)
}

fn build_schema(embedding_dim: i32) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("record_id", DataType::Utf8, false),
        Field::new("path", DataType::Utf8, false),
        Field::new("file_name", DataType::Utf8, false),
        Field::new("extension", DataType::Utf8, false),
        Field::new("parent_dir", DataType::Utf8, false),
        Field::new("modality", DataType::Utf8, false),
        Field::new("chunk", DataType::UInt32, false),
        Field::new("offset", DataType::UInt64, false),
        Field::new("plaintext", DataType::Utf8, false),
        Field::new("size_bytes", DataType::UInt64, false),
        Field::new("modified_at", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                embedding_dim,
            ),
            true,
        ),
    ]))
}

fn build_batch(records: &[WorkerRecord], embedding_dim: i32) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let schema = build_schema(embedding_dim);
    let record_ids = StringArray::from_iter_values(
        records
            .iter()
            .map(|record| record.record_id.as_deref().unwrap_or("")),
    );
    let paths = StringArray::from_iter_values(records.iter().map(|record| record.path.as_str()));
    let file_names = StringArray::from_iter_values(
        records
            .iter()
            .map(|record| record.file_name.as_deref().unwrap_or("")),
    );
    let extensions = StringArray::from_iter_values(
        records
            .iter()
            .map(|record| record.extension.as_deref().unwrap_or("")),
    );
    let parent_dirs = StringArray::from_iter_values(
        records
            .iter()
            .map(|record| record.parent_dir.as_deref().unwrap_or("")),
    );
    let modalities = StringArray::from_iter_values(
        records
            .iter()
            .map(|record| record.modality.as_deref().unwrap_or("")),
    );
    let chunks = UInt32Array::from_iter_values(records.iter().map(|record| record.chunk.unwrap_or(0)));
    let offsets = UInt64Array::from_iter_values(records.iter().map(|record| record.offset.unwrap_or(0)));
    let plaintexts = StringArray::from_iter_values(
        records
            .iter()
            .map(|record| record.plaintext.as_deref().unwrap_or("")),
    );
    let sizes = UInt64Array::from_iter_values(records.iter().map(|record| record.size_bytes.unwrap_or(0)));
    let modified_ats = StringArray::from_iter_values(
        records
            .iter()
            .map(|record| record.modified_at.as_deref().unwrap_or("")),
    );
    let embeddings = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        records.iter().map(|record| {
            let vector = record.embedding.as_ref().expect("missing embedding for stored record");
            Some(vector.iter().copied().map(Some).collect::<Vec<_>>())
        }),
        embedding_dim,
    );

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(record_ids) as ArrayRef,
            Arc::new(paths) as ArrayRef,
            Arc::new(file_names),
            Arc::new(extensions),
            Arc::new(parent_dirs),
            Arc::new(modalities),
            Arc::new(chunks),
            Arc::new(offsets),
            Arc::new(plaintexts),
            Arc::new(sizes),
            Arc::new(modified_ats),
            Arc::new(embeddings),
        ],
    )?)
}

// Source of truth for document extensions passed to the Python worker.
const DOCUMENT_EXTENSIONS: &[&str] = &[
    ".csv",
    ".dbf",
    ".dif",
    ".doc",
    ".docm",
    ".docx",
    ".dot",
    ".dotm",
    ".dotx",
    ".fodg",
    ".fodp",
    ".fods",
    ".fodt",
    ".htm",
    ".html",
    ".mht",
    ".mhtml",
    ".odb",
    ".odc",
    ".odf",
    ".odg",
    ".odm",
    ".odp",
    ".ods",
    ".odt",
    ".oth",
    ".otp",
    ".ots",
    ".ott",
    ".otg",
    ".otm",
    ".pot",
    ".potm",
    ".potx",
    ".pps",
    ".ppsm",
    ".ppsx",
    ".ppt",
    ".pptm",
    ".pptx",
    ".rtf",
    ".sda",
    ".sdc",
    ".sdd",
    ".sdw",
    ".slk",
    ".sxc",
    ".sxd",
    ".sxg",
    ".sxi",
    ".sxm",
    ".sxw",
    ".tab",
    ".tsv",
    ".txt",
    ".uot",
    ".uop",
    ".uos",
    ".uof",
    ".vdx",
    ".vsd",
    ".vsdx",
    ".svg",
    ".xhtml",
    ".xls",
    ".xlsm",
    ".xlsx",
    ".xlt",
    ".xltm",
    ".xltx",
    ".xml",
];

fn is_document_extension(extension: &str) -> bool {
    DOCUMENT_EXTENSIONS.contains(&extension)
}

fn document_extensions_arg() -> String {
    DOCUMENT_EXTENSIONS.join(",")
}

fn snippet_unit(modality: &str, extension: &str) -> &'static str {
    if extension == ".pdf" || is_document_extension(extension) {
        return "page";
    }
    if matches!(extension, ".3gp" | ".mp4" | ".m2ts" | ".mkv" | ".mov" | ".avi" | ".m4v" | ".mpeg" | ".mpg" | ".ts" | ".webm") {
        return "frame";
    }
    if modality == "audio" || matches!(extension, ".aac" | ".aif" | ".aiff" | ".au" | ".flac" | ".m4a" | ".mp3" | ".ogg" | ".opus" | ".wav") {
        return "second";
    }
    if modality == "text" {
        return "byte";
    }
    "offset"
}

fn display_offset(extension: &str, offset: u64) -> u64 {
    if extension == ".pdf" || is_document_extension(extension) {
        offset + 1
    } else {
        offset
    }
}

async fn open_table(connection: &Connection, table_name: &str) -> Option<Table> {
    connection.open_table(table_name).execute().await.ok()
}

async fn flush_records(
    connection: &Connection,
    table_name: &str,
    table: &mut Option<Table>,
    pending: &mut Vec<WorkerRecord>,
    embedding_dim: &mut Option<i32>,
) -> Result<usize, Box<dyn std::error::Error>> {
    if pending.is_empty() {
        return Ok(0);
    }

    let current_dim = pending[0]
        .embedding
        .as_ref()
        .ok_or("missing embedding in pending record")?
        .len() as i32;

    match embedding_dim {
        Some(existing_dim) if *existing_dim != current_dim => {
            return Err(
                format!("embedding dimension changed from {existing_dim} to {current_dim}").into(),
            )
        }
        None => *embedding_dim = Some(current_dim),
        _ => {}
    }

    let batch = build_batch(pending, current_dim)?;
    let schema = batch.schema();
    let batches = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
    let batch_reader: Box<dyn arrow_array::RecordBatchReader + Send> = Box::new(batches);

    if table.is_none() {
        *table = open_table(connection, table_name).await;
    }

    if let Some(existing_table) = table.as_ref() {
        let mut merge = existing_table.merge_insert(&["record_id"]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        merge.execute(batch_reader).await?;
    } else {
        *table = Some(
            connection
                .create_table(table_name, batch_reader)
                .execute()
                .await?,
        );
    }

    let flushed = pending.len();
    pending.clear();
    Ok(flushed)
}

fn normalize_existing_path(path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if path.exists() {
        Ok(fs::canonicalize(path)?)
    } else if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn sql_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

async fn delete_path_records(
    connection: &Connection,
    table_name: &str,
    table: &mut Option<Table>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if table.is_none() {
        *table = open_table(connection, table_name).await;
    }

    if let Some(existing_table) = table.as_ref() {
        existing_table
            .delete(&format!("path = {}", sql_quote(path)))
            .await?;
    }

    Ok(())
}

fn start_worker(args: &Args) -> Result<WorkerSession, Box<dyn std::error::Error>> {
    let mut child = Command::new(&args.python)
        .arg(&args.script)
        .arg("--model-dir")
        .arg(&args.model_dir)
        .arg("--task")
        .arg(&args.task)
        .arg("--device")
        .arg(&args.device)
        .arg("--document-extensions")
        .arg(document_extensions_arg())
        .args(if args.translate { vec!["--translate"] } else { Vec::new() })
        .args(
            args.qwen_max_memory
                .map(|value| vec!["--qwen-max-memory".to_string(), value.to_string()])
                .unwrap_or_default(),
        )
        .args(if args.low_memory { vec!["--low-memory"] } else { Vec::new() })
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;

    let stdin = child.stdin.take().ok_or("failed to open embedding worker stdin")?;
    let stdout = child.stdout.take().ok_or("failed to open embedding worker stdout")?;

    Ok(WorkerSession {
        child,
        writer: BufWriter::new(stdin),
        reader: BufReader::new(stdout),
    })
}

fn send_worker_file(session: &mut WorkerSession, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let request = json!({
        "type": "file",
        "path": path.to_string_lossy(),
    });
    writeln!(session.writer, "{request}")?;
    session.writer.flush()?;
    Ok(())
}

fn shutdown_worker(session: &mut WorkerSession) -> Result<(), Box<dyn std::error::Error>> {
    let request = json!({ "type": "shutdown" });
    writeln!(session.writer, "{request}")?;
    session.writer.flush()?;
    Ok(())
}

async fn ingest_single_path(
    session: &mut WorkerSession,
    connection: &Connection,
    table_name: &str,
    table: &mut Option<Table>,
    path: &Path,
    embedding_dim: &mut Option<i32>,
) -> Result<Summary, Box<dyn std::error::Error>> {
    let normalized_path = normalize_existing_path(path)?;
    let normalized_path_str = normalized_path.to_string_lossy().to_string();
    delete_path_records(connection, table_name, table, &normalized_path_str).await?;
    send_worker_file(session, &normalized_path)?;

    let mut pending = Vec::new();
    let mut summary = Summary {
        stored: 0,
        skipped: 0,
        errors: 0,
    };

    loop {
        let mut line = String::new();
        if session.reader.read_line(&mut line)? == 0 {
            return Err("embedding worker exited unexpectedly".into());
        }
        if line.trim().is_empty() {
            continue;
        }

        let value: serde_json::Value = serde_json::from_str(&line)?;
        let status = value
            .get("status")
            .and_then(|status| status.as_str())
            .ok_or("worker response missing status")?;

        match status {
            "ok" => {
                let record: WorkerRecord = serde_json::from_value(value)?;
                pending.push(record);
            }
            "skipped" => {
                summary.skipped += 1;
            }
            "error" => {
                let record: WorkerRecord = serde_json::from_value(value)?;
                summary.errors += 1;
                let reason = record.reason.unwrap_or_else(|| "unknown error".to_string());
                eprintln!("{}: {}", record.path, reason);
            }
            "done" => {
                break;
            }
            other => return Err(format!("unknown worker status: {other}").into()),
        }
    }

    summary.stored += flush_records(connection, table_name, table, &mut pending, embedding_dim).await?;
    Ok(summary)
}

fn should_ignore_event_path(
    path: &Path,
    watched_root: &Path,
    target_file: Option<&Path>,
    table_target: &TableTarget,
    ignore_matcher: &IgnoreMatcher,
) -> bool {
    if path.starts_with(&table_target.table_path) {
        return true;
    }
    if should_ignore_path(path, ignore_matcher) {
        return true;
    }
    if let Some(target_file) = target_file {
        return path != target_file;
    }
    !path.starts_with(watched_root)
}

fn classify_event(event: &Event) -> Option<bool> {
    if event.kind.is_remove() {
        Some(false)
    } else if event.kind.is_create() || event.kind.is_modify() {
        Some(true)
    } else {
        None
    }
}

async fn run_ingest(
    args: &Args,
    connection: &Connection,
    table_target: &TableTarget,
    root_path: &Path,
    ignore_matcher: &IgnoreMatcher,
) -> Result<(), Box<dyn std::error::Error>> {
    let files = collect_files(root_path, ignore_matcher)?;
    let total_files = files.len();
    let mut session = start_worker(args)?;
    let mut table = open_table(connection, &table_target.table_name).await;
    let mut embedding_dim = None;
    let mut summary = Summary {
        stored: 0,
        skipped: 0,
        errors: 0,
    };

    for (index, file) in files.into_iter().enumerate() {
        eprintln!(
            "ingest {}/{}: {}",
            index + 1,
            total_files,
            file.display()
        );
        let file_summary = ingest_single_path(
            &mut session,
            connection,
            &table_target.table_name,
            &mut table,
            &file,
            &mut embedding_dim,
        )
        .await?;
        summary.stored += file_summary.stored;
        summary.skipped += file_summary.skipped;
        summary.errors += file_summary.errors;
    }

    shutdown_worker(&mut session)?;
    let status = session.child.wait()?;
    if !status.success() {
        return Err("embedding script failed".into());
    }

    println!(
        "{}",
        json!({
            "db": table_target.table_path.to_string_lossy(),
            "table": table_target.table_name,
            "stored": summary.stored,
            "skipped": summary.skipped,
            "errors": summary.errors,
        })
    );

    Ok(())
}

async fn run_watch(
    args: &Args,
    connection: &Connection,
    table_target: &TableTarget,
    root_path: &Path,
    ignore_matcher: &IgnoreMatcher,
) -> Result<(), Box<dyn std::error::Error>> {
    run_ingest(args, connection, table_target, root_path, ignore_matcher).await?;

    let watch_target = normalize_existing_path(root_path)?;
    let watched_root = if watch_target.is_dir() {
        watch_target.clone()
    } else {
        watch_target.parent().unwrap_or(Path::new(".")).to_path_buf()
    };
    let target_file = if watch_target.is_file() {
        Some(watch_target.clone())
    } else {
        None
    };

    let mut session = start_worker(args)?;
    let mut table = open_table(connection, &table_target.table_name).await;
    let mut embedding_dim = None;
    let (tx, rx) = channel();
    let mut watcher = recommended_watcher(move |res| {
        let _ = tx.send(res);
    })?;
    watcher.watch(
        &watched_root,
        if watch_target.is_dir() {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        },
    )?;

    loop {
        match rx.recv() {
            Ok(Ok(event)) => {
                let Some(should_index) = classify_event(&event) else {
                    continue;
                };

                for path in event.paths {
                    if should_ignore_event_path(
                        &path,
                        &watched_root,
                        target_file.as_deref(),
                        table_target,
                        ignore_matcher,
                    ) {
                        continue;
                    }

                    if should_index {
                        if path.is_file() {
                            let summary = ingest_single_path(
                                &mut session,
                                connection,
                                &table_target.table_name,
                                &mut table,
                                &path,
                                &mut embedding_dim,
                            )
                            .await?;
                            if summary.errors > 0 {
                                eprintln!(
                                    "reindex errors for {}: {}",
                                    path.display(),
                                    summary.errors
                                );
                            }
                        }
                    } else {
                        let normalized = normalize_existing_path(&path)?;
                        delete_path_records(
                            connection,
                            &table_target.table_name,
                            &mut table,
                            &normalized.to_string_lossy(),
                        )
                        .await?;
                    }
                }
            }
            Ok(Err(err)) => eprintln!("watch error: {err}"),
            Err(err) => return Err(format!("watch channel closed: {err}").into()),
        }
    }
}

async fn read_query_embedding(
    args: &Args,
    query: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut child = Command::new(&args.python)
        .arg(&args.script)
        .arg("--model-dir")
        .arg(&args.model_dir)
        .arg("--task")
        .arg(&args.task)
        .arg("--device")
        .arg(&args.device)
        .arg("--query-text")
        .arg(query)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;

    let stdout = child.stdout.take().ok_or("failed to open query worker stdout")?;
    let reader = BufReader::new(stdout);
    let mut embedding = None;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: QueryRecord = serde_json::from_str(&line)?;
        match record.status.as_str() {
            "query" => embedding = record.embedding,
            "error" => {
                let reason = record.reason.unwrap_or_else(|| "unknown error".to_string());
                return Err(format!("query embedding failed: {reason}").into());
            }
            other => return Err(format!("unknown worker status: {other}").into()),
        }
    }

    let status = child.wait()?;
    if !status.success() {
        return Err("query embedding script failed".into());
    }

    embedding.ok_or_else(|| "query worker did not return an embedding".into())
}

async fn run_search(
    args: &Args,
    connection: &Connection,
    table_target: &TableTarget,
    query: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let (range_start, range_end) = if let Some(range_value) = args.range.as_deref() {
        let (start, end) = parse_range(range_value).map_err(|err| format!("invalid --range: {err}"))?;
        (start, Some(end))
    } else {
        (0, None)
    };
    let effective_limit = range_end.unwrap_or(args.limit);

    let table = open_table(connection, &table_target.table_name)
        .await
        .ok_or("search table does not exist")?;
    let query_vector = read_query_embedding(args, query).await?;
    let mut results = table
        .vector_search(query_vector)?
        .limit(effective_limit)
        .execute()
        .await?;

    let mut global_index = 0usize;
    let mut json_rows: Vec<serde_json::Value> = Vec::new();
    while let Some(batch) = results.next().await {
        let batch = batch?;
        let paths = batch
            .column_by_name("path")
            .ok_or("search result missing path column")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("path column is not Utf8")?;
        let file_names = batch
            .column_by_name("file_name")
            .ok_or("search result missing file_name column")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("file_name column is not Utf8")?;
        let extensions = batch
            .column_by_name("extension")
            .ok_or("search result missing extension column")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("extension column is not Utf8")?;
        let modalities = batch
            .column_by_name("modality")
            .ok_or("search result missing modality column")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("modality column is not Utf8")?;
        let offsets = batch
            .column_by_name("offset")
            .and_then(|column| column.as_any().downcast_ref::<UInt64Array>());
        let plaintexts = batch
            .column_by_name("plaintext")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>());

        for index in 0..batch.num_rows() {
            if global_index < range_start {
                global_index += 1;
                continue;
            }
            if let Some(end) = range_end && global_index >= end {
                return Ok(());
            }
            let path = paths.value(index);
            let file_name = file_names.value(index);
            let modality = modalities.value(index);
            let extension = extensions.value(index);
            let offset = offsets.map(|array| array.value(index)).unwrap_or(0);
            let unit = snippet_unit(modality, extension);
            let resolved_offset = display_offset(extension, offset);
            let snippet_text = plaintexts
                .map(|array| array.value(index))
                .unwrap_or("")
                .replace(['\t', '\n'], " ");
            if args.json {
                json_rows.push(json!({
                    "path": path,
                    "file_name": file_name,
                    "modality": modality,
                    "unit": unit,
                    "offset": resolved_offset,
                    "snippet": snippet_text,
                }));
            } else {
                println!("{path}\t{file_name}\t{modality}\t{unit}:{resolved_offset}\t{snippet_text}");
            }
            global_index += 1;
        }
    }

    if args.json {
        println!("{}", serde_json::to_string(&json_rows)?);
    }

    Ok(())
}

fn collect_files(path: &Path, ignore_matcher: &IgnoreMatcher) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    if should_ignore_path(path, ignore_matcher) {
        return Ok(Vec::new());
    }

    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }

    if !path.is_dir() {
        return Err(format!("path does not exist or is not accessible: {}", path.display()).into());
    }

    let mut files = Vec::new();
    collect_files_recursive(path, ignore_matcher, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_files_recursive(
    path: &Path,
    ignore_matcher: &IgnoreMatcher,
    files: &mut Vec<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();
        if should_ignore_path(&entry_path, ignore_matcher) {
            continue;
        }
        let file_type = entry.file_type()?;

        if file_type.is_dir() {
            collect_files_recursive(&entry_path, ignore_matcher, files)?;
        } else if file_type.is_file() {
            files.push(entry_path);
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let table_target = resolve_table_target(&args.db);
    let ignore_matcher = load_ignore_matcher(&args)?;
    let mode = match (args.path.as_deref(), args.search.as_deref()) {
        (Some(path), None) => Mode::Ingest(path),
        (None, Some(query)) => Mode::Search(query),
        (Some(_), Some(_)) => return Err("use either --path or --search, not both".into()),
        (None, None) => return Err("either --path or --search is required".into()),
    };

    if matches!(mode, Mode::Ingest(_)) {
        fs::create_dir_all(&table_target.db_root)?;
    }

    let connection = connect(table_target.db_root.to_string_lossy().as_ref())
        .execute()
        .await?;

    if let Mode::Search(query) = mode {
        return run_search(&args, &connection, &table_target, query).await;
    }

    let root_path = match mode {
        Mode::Ingest(path) => path,
        Mode::Search(_) => unreachable!(),
    };

    if args.watch {
        run_watch(&args, &connection, &table_target, root_path, &ignore_matcher).await
    } else {
        run_ingest(&args, &connection, &table_target, root_path, &ignore_matcher).await
    }
}

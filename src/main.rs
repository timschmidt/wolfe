use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{ArrayRef, FixedSizeListArray, RecordBatch, RecordBatchIterator, StringArray, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use clap::Parser;
use futures::StreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, Table, connect};
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
}

#[derive(Debug, Deserialize)]
struct WorkerRecord {
    status: String,
    record_id: Option<String>,
    path: String,
    file_name: Option<String>,
    extension: Option<String>,
    parent_dir: Option<String>,
    modality: Option<String>,
    chunk: Option<u32>,
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

fn build_schema(embedding_dim: i32) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("record_id", DataType::Utf8, false),
        Field::new("path", DataType::Utf8, false),
        Field::new("file_name", DataType::Utf8, false),
        Field::new("extension", DataType::Utf8, false),
        Field::new("parent_dir", DataType::Utf8, false),
        Field::new("modality", DataType::Utf8, false),
        Field::new("chunk", DataType::UInt32, false),
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
            Arc::new(sizes),
            Arc::new(modified_ats),
            Arc::new(embeddings),
        ],
    )?)
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
    let table = open_table(connection, &table_target.table_name)
        .await
        .ok_or("search table does not exist")?;
    let query_vector = read_query_embedding(args, query).await?;
    let mut results = table
        .vector_search(query_vector)?
        .limit(args.limit)
        .execute()
        .await?;

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

        for index in 0..batch.num_rows() {
            let path = paths.value(index);
            let file_name = file_names.value(index);
            println!("{path}\t{file_name}");
        }
    }

    Ok(())
}

fn collect_files(path: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }

    if !path.is_dir() {
        return Err(format!("path does not exist or is not accessible: {}", path.display()).into());
    }

    let mut files = Vec::new();
    collect_files_recursive(path, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_files_recursive(
    path: &Path,
    files: &mut Vec<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();
        let file_type = entry.file_type()?;

        if file_type.is_dir() {
            collect_files_recursive(&entry_path, files)?;
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

    let files = collect_files(match mode {
        Mode::Ingest(path) => path,
        Mode::Search(_) => unreachable!(),
    })?;

    let mut child = Command::new(&args.python)
        .arg(&args.script)
        .arg("--model-dir")
        .arg(&args.model_dir)
        .arg("--task")
        .arg(&args.task)
        .arg("--device")
        .arg(&args.device)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;

    {
        let stdin = child.stdin.take().ok_or("failed to open embedding worker stdin")?;
        let mut writer = BufWriter::new(stdin);

        for file in files {
            writer.write_all(file.to_string_lossy().as_bytes())?;
            writer.write_all(b"\n")?;
        }

        writer.flush()?;
    }

    let mut table = open_table(&connection, &table_target.table_name).await;
    let mut pending = Vec::new();
    let mut embedding_dim = None;
    let mut summary = Summary {
        stored: 0,
        skipped: 0,
        errors: 0,
    };
    let stdout = child.stdout.take().ok_or("failed to open embedding worker stdout")?;
    let reader = BufReader::new(stdout);

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let record: WorkerRecord = serde_json::from_str(&line)?;
        match record.status.as_str() {
            "ok" => {
                pending.push(record);
                if pending.len() >= 64 {
                    summary.stored += flush_records(
                        &connection,
                        &table_target.table_name,
                        &mut table,
                        &mut pending,
                        &mut embedding_dim,
                    )
                    .await?;
                }
            }
            "skipped" => {
                summary.skipped += 1;
            }
            "error" => {
                summary.errors += 1;
                let reason = record.reason.unwrap_or_else(|| "unknown error".to_string());
                eprintln!("{}: {}", record.path, reason);
            }
            other => {
                return Err(format!("unknown worker status: {other}").into());
            }
        }
    }

    summary.stored += flush_records(
        &connection,
        &table_target.table_name,
        &mut table,
        &mut pending,
        &mut embedding_dim,
    )
    .await?;

    let status = child.wait()?;

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

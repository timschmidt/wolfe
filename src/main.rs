use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::net::SocketAddr;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::mpsc::channel;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    env,
};

use arrow_array::types::Float32Type;
use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, Float64Array, RecordBatch, RecordBatchIterator,
    StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use axum::{Json, Router};
use clap::Parser;
use futures::StreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, Table, connect};
use notify::{Event, RecursiveMode, Watcher, recommended_watcher};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to a file or directory to embed recursively
    #[arg(
        short,
        long,
        value_name = "PATH",
        alias = "file",
        conflicts_with = "search"
    )]
    path: Option<PathBuf>,

    /// File containing newline-separated files to embed
    #[arg(long, value_name = "FILE", conflicts_with_all = ["path", "search", "download_models", "watch", "web", "match_context", "enrich_corpus"])]
    path_list: Option<PathBuf>,

    /// Query string to vectorize and search semantically
    #[arg(long, value_name = "TEXT", conflicts_with_all = ["path", "path_list", "download_models", "match_context"])]
    search: Option<String>,

    /// Lexical term to find and print merged neighboring text chunks for each source
    #[arg(long, value_name = "TEXT", conflicts_with_all = ["path", "path_list", "search", "download_models", "watch", "web", "enrich_corpus"])]
    match_context: Option<String>,

    /// Number of chunks before and after each lexical match to include
    #[arg(long, default_value_t = 1)]
    context_window: u32,

    /// Embedding task name
    #[arg(long, default_value = "retrieval")]
    task: String,

    /// Path to the Lance table directory
    #[arg(long, default_value = "wolfe.lance")]
    db: PathBuf,

    /// Path to the Python interpreter
    #[arg(long, default_value = "python3")]
    python: String,

    /// Execution device: auto, cpu, cuda, cuda:N, or mps
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

    /// Build per-source enrichment metadata from an existing Wolfe index
    #[arg(long, conflicts_with_all = ["path", "path_list", "search", "match_context", "download_models", "watch", "web"])]
    enrich_corpus: bool,

    /// Enrichment pass to run: source-metadata, references, concepts, or all
    #[arg(long, default_value = "source-metadata", requires = "enrich_corpus")]
    enrichment_pass: String,

    /// Directory for per-source enrichment sidecar JSON files
    #[arg(
        long,
        value_name = "DIR",
        default_value = "wolfe-metadata",
        requires = "enrich_corpus"
    )]
    metadata_root: PathBuf,

    /// JSONL catalog path for source enrichment metadata
    #[arg(
        long,
        value_name = "FILE",
        default_value = "wolfe-metadata/catalog.jsonl",
        requires = "enrich_corpus"
    )]
    metadata_catalog: PathBuf,

    /// JSONL catalog path for reference and citation candidates
    #[arg(
        long,
        value_name = "FILE",
        default_value = "wolfe-metadata/references.jsonl",
        requires = "enrich_corpus"
    )]
    reference_catalog: PathBuf,

    /// JSONL catalog path for concept and definition candidates
    #[arg(
        long,
        value_name = "FILE",
        default_value = "wolfe-metadata/concepts.jsonl",
        requires = "enrich_corpus"
    )]
    concept_catalog: PathBuf,

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

    /// Pre-download Jina, Whisper, Qwen, and YAMNet into their normal cache directories
    #[arg(long, conflicts_with_all = ["path", "search", "watch"])]
    download_models: bool,

    /// Run the local web search UI instead of ingesting or running one CLI search
    #[arg(long, conflicts_with_all = ["path", "search", "download_models", "watch"])]
    web: bool,

    /// Listen address for --web
    #[arg(long, default_value = "127.0.0.1:8767", requires = "web")]
    listen: SocketAddr,

    /// OpenAI-compatible embeddings endpoint to use for query embeddings
    #[arg(long, value_name = "URL")]
    embedding_url: Option<String>,

    /// Model name to send to --embedding-url
    #[arg(long, default_value = "wolfe-jina")]
    embedding_model: String,

    /// API key to send as Authorization Bearer and X-Api-Key for --embedding-url
    #[arg(long, env = "WOLFE_EMBEDDING_API_KEY")]
    embedding_api_key: Option<String>,
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
    IngestList(&'a Path),
    Search(&'a str),
    MatchContext(&'a str),
    EnrichCorpus,
    Web,
}

#[derive(Clone)]
struct WebState {
    db_root: PathBuf,
    table_name: String,
    query_config: QueryConfig,
}

#[derive(Clone)]
struct QueryConfig {
    python: String,
    script: PathBuf,
    task: String,
    device: String,
    embedding_url: Option<String>,
    embedding_model: String,
    embedding_api_key: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct SearchParams {
    q: String,
    limit: Option<usize>,
    offset: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ContextParams {
    path: String,
    chunk: Option<u32>,
    offset: Option<u64>,
    window: Option<u32>,
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    query: String,
    limit: usize,
    offset: usize,
    results: Vec<SearchRow>,
}

#[derive(Debug, Serialize)]
struct ContextResponse {
    path: String,
    window: u32,
    results: Vec<SearchRow>,
}

#[derive(Debug, Clone, Serialize)]
struct SearchRow {
    record_id: String,
    path: String,
    file_name: String,
    extension: String,
    parent_dir: String,
    modality: String,
    chunk: u32,
    offset: u64,
    display_offset: u64,
    unit: &'static str,
    snippet: String,
    size_bytes: u64,
    modified_at: String,
    distance: Option<f64>,
}

#[derive(Debug, Serialize)]
struct MatchContextResponse {
    term: String,
    window: u32,
    sources: Vec<MatchContextSource>,
}

#[derive(Debug, Serialize)]
struct MatchContextSource {
    path: String,
    file_name: String,
    extension: String,
    parent_dir: String,
    matches: usize,
    windows: Vec<MatchContextWindow>,
}

#[derive(Debug, Serialize)]
struct MatchContextWindow {
    start_chunk: u32,
    end_chunk: u32,
    text: String,
    chunks: Vec<SearchRow>,
}

#[derive(Debug, Serialize)]
struct CorpusEnrichmentSummary {
    pass: String,
    sources: usize,
    reference_candidates: usize,
    concept_candidates: usize,
    sidecar_root: String,
    catalog: String,
    reference_catalog: String,
    concept_catalog: String,
}

#[derive(Debug, Serialize)]
struct SourceMetadata {
    schema_version: u32,
    source_id: String,
    path: String,
    file_name: String,
    extension: String,
    parent_dir: String,
    document_type: String,
    title_guess: Option<String>,
    collection_path: Vec<String>,
    source_size_bytes: u64,
    file_size_bytes: Option<u64>,
    file_modified_unix: Option<u64>,
    indexed_modified_at: String,
    row_count: usize,
    text_chunk_count: usize,
    image_chunk_count: usize,
    other_chunk_count: usize,
    chunk_count: usize,
    page_count: Option<u64>,
    text_char_count: usize,
    word_count: usize,
    modalities: BTreeMap<String, usize>,
    bibtex_candidate: BibtexCandidate,
    enrichment_run: EnrichmentRunMetadata,
}

#[derive(Debug, Serialize)]
struct BibtexCandidate {
    entry_type: String,
    citation_key: String,
    title: Option<String>,
    year: Option<String>,
    howpublished: String,
    note: String,
    confidence: f64,
    sources: Vec<String>,
    bibtex: String,
}

#[derive(Debug, Serialize)]
struct EnrichmentRunMetadata {
    tool: String,
    pass: String,
    generated_unix: u64,
}

#[derive(Debug, Serialize)]
struct ReferenceCandidate {
    schema_version: u32,
    candidate_id: String,
    source_id: String,
    path: String,
    file_name: String,
    extension: String,
    parent_dir: String,
    chunk: u32,
    offset: u64,
    display_offset: u64,
    unit: &'static str,
    kind: String,
    text: String,
    doi: Option<String>,
    url: Option<String>,
    year: Option<String>,
    confidence: f64,
    enrichment_run: EnrichmentRunMetadata,
}

#[derive(Debug, Serialize)]
struct ConceptCandidate {
    schema_version: u32,
    candidate_id: String,
    source_id: String,
    path: String,
    file_name: String,
    extension: String,
    parent_dir: String,
    chunk: u32,
    offset: u64,
    display_offset: u64,
    unit: &'static str,
    phrase: String,
    normalized_phrase: String,
    definition_type: String,
    evidence_text: String,
    confidence: f64,
    enrichment_run: EnrichmentRunMetadata,
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

fn load_path_list(path_list: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path_list)?;
    let mut paths = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        paths.push(PathBuf::from(trimmed));
    }
    Ok(paths)
}

fn path_matches_ignore_name(path: &Path, matcher: &IgnoreMatcher) -> bool {
    path.components().any(|component| match component {
        Component::Normal(name) => matcher.names.contains(&name.to_string_lossy().to_string()),
        _ => false,
    })
}

fn path_matches_ignore_path(path: &Path, matcher: &IgnoreMatcher) -> bool {
    matcher
        .paths
        .iter()
        .any(|ignored| path.starts_with(ignored))
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

fn build_batch(
    records: &[WorkerRecord],
    embedding_dim: i32,
) -> Result<RecordBatch, Box<dyn std::error::Error>> {
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
    let chunks =
        UInt32Array::from_iter_values(records.iter().map(|record| record.chunk.unwrap_or(0)));
    let offsets =
        UInt64Array::from_iter_values(records.iter().map(|record| record.offset.unwrap_or(0)));
    let plaintexts = StringArray::from_iter_values(
        records
            .iter()
            .map(|record| record.plaintext.as_deref().unwrap_or("")),
    );
    let sizes =
        UInt64Array::from_iter_values(records.iter().map(|record| record.size_bytes.unwrap_or(0)));
    let modified_ats = StringArray::from_iter_values(
        records
            .iter()
            .map(|record| record.modified_at.as_deref().unwrap_or("")),
    );
    let embeddings = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        records.iter().map(|record| {
            let vector = record
                .embedding
                .as_ref()
                .expect("missing embedding for stored record");
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
    ".csv", ".dbf", ".dif", ".doc", ".docm", ".docx", ".dot", ".dotm", ".dotx", ".fodg", ".fodp",
    ".fods", ".fodt", ".htm", ".html", ".mht", ".mhtml", ".odb", ".odc", ".odf", ".odg", ".odm",
    ".odp", ".ods", ".odt", ".oth", ".otp", ".ots", ".ott", ".otg", ".otm", ".pot", ".potm",
    ".potx", ".pps", ".ppsm", ".ppsx", ".ppt", ".pptm", ".pptx", ".rtf", ".sda", ".sdc", ".sdd",
    ".sdw", ".slk", ".sxc", ".sxd", ".sxg", ".sxi", ".sxm", ".sxw", ".tab", ".tsv", ".txt", ".uot",
    ".uop", ".uos", ".uof", ".vdx", ".vsd", ".vsdx", ".wp", ".wp5", ".wp6", ".wpd", ".svg",
    ".xhtml", ".xls", ".xlsm", ".xlsx", ".xlt", ".xltm", ".xltx", ".xml",
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
    if matches!(
        extension,
        ".3gp"
            | ".mp4"
            | ".m2ts"
            | ".mkv"
            | ".mov"
            | ".avi"
            | ".m4v"
            | ".mpeg"
            | ".mpg"
            | ".ts"
            | ".webm"
    ) {
        return "frame";
    }
    if modality == "audio"
        || matches!(
            extension,
            ".aac"
                | ".aif"
                | ".aiff"
                | ".au"
                | ".flac"
                | ".m4a"
                | ".mp3"
                | ".ogg"
                | ".opus"
                | ".wav"
        )
    {
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

fn query_config_from_args(args: &Args) -> QueryConfig {
    QueryConfig {
        python: args.python.clone(),
        script: args.script.clone(),
        task: args.task.clone(),
        device: args.device.clone(),
        embedding_url: args.embedding_url.clone(),
        embedding_model: args.embedding_model.clone(),
        embedding_api_key: args.embedding_api_key.clone(),
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
            return Err(format!(
                "embedding dimension changed from {existing_dim} to {current_dim}"
            )
            .into());
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
        .arg("--task")
        .arg(&args.task)
        .arg("--device")
        .arg(&args.device)
        .arg("--document-extensions")
        .arg(document_extensions_arg())
        .args(if args.translate {
            vec!["--translate"]
        } else {
            Vec::new()
        })
        .args(
            args.qwen_max_memory
                .map(|value| vec!["--qwen-max-memory".to_string(), value.to_string()])
                .unwrap_or_default(),
        )
        .args(if args.low_memory {
            vec!["--low-memory"]
        } else {
            Vec::new()
        })
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;

    let stdin = child
        .stdin
        .take()
        .ok_or("failed to open embedding worker stdin")?;
    let stdout = child
        .stdout
        .take()
        .ok_or("failed to open embedding worker stdout")?;

    Ok(WorkerSession {
        child,
        writer: BufWriter::new(stdin),
        reader: BufReader::new(stdout),
    })
}

fn download_models(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let status = Command::new(&args.python)
        .arg(&args.script)
        .arg("--download-models")
        .arg("--device")
        .arg(&args.device)
        .stderr(Stdio::inherit())
        .stdout(Stdio::inherit())
        .status()?;

    if !status.success() {
        return Err("model download script failed".into());
    }

    Ok(())
}

fn send_worker_file(
    session: &mut WorkerSession,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
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

    summary.stored +=
        flush_records(connection, table_name, table, &mut pending, embedding_dim).await?;
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
    run_ingest_file_set(args, connection, table_target, files).await
}

async fn run_ingest_file_set(
    args: &Args,
    connection: &Connection,
    table_target: &TableTarget,
    files: Vec<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
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
        eprintln!("ingest {}/{}: {}", index + 1, total_files, file.display());
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

async fn run_ingest_list(
    args: &Args,
    connection: &Connection,
    table_target: &TableTarget,
    path_list: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let files = load_path_list(path_list)?;
    run_ingest_file_set(args, connection, table_target, files).await
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
        watch_target
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf()
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
    config: &QueryConfig,
    query: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if let Some(embedding_url) = &config.embedding_url {
        let client = reqwest::Client::new();
        let mut request = client.post(embedding_url).json(&json!({
            "model": config.embedding_model,
            "input": query,
            "task": config.task,
        }));
        if let Some(api_key) = &config.embedding_api_key {
            request = request.bearer_auth(api_key).header("X-Api-Key", api_key);
        }
        let response = request.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("embedding endpoint returned {status}: {body}").into());
        }
        let payload: OpenAiEmbeddingResponse = response.json().await?;
        return payload
            .data
            .into_iter()
            .next()
            .map(|row| row.embedding)
            .ok_or_else(|| "embedding endpoint returned no embeddings".into());
    }

    let mut child = Command::new(&config.python)
        .arg(&config.script)
        .arg("--task")
        .arg(&config.task)
        .arg("--device")
        .arg(&config.device)
        .arg("--query-text")
        .arg(query)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;

    let stdout = child
        .stdout
        .take()
        .ok_or("failed to open query worker stdout")?;
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

fn column_string(
    batch: &RecordBatch,
    name: &str,
    index: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let array = batch
        .column_by_name(name)
        .ok_or_else(|| format!("search result missing {name} column"))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| format!("{name} column is not Utf8"))?;
    Ok(array.value(index).to_string())
}

fn column_u32(batch: &RecordBatch, name: &str, index: usize) -> u32 {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<UInt32Array>())
        .map(|array| array.value(index))
        .unwrap_or(0)
}

fn column_u64(batch: &RecordBatch, name: &str, index: usize) -> u64 {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
        .map(|array| array.value(index))
        .unwrap_or(0)
}

fn column_distance(batch: &RecordBatch, index: usize) -> Option<f64> {
    if let Some(array) = batch
        .column_by_name("_distance")
        .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
    {
        return Some(array.value(index) as f64);
    }
    batch
        .column_by_name("_distance")
        .and_then(|column| column.as_any().downcast_ref::<Float64Array>())
        .map(|array| array.value(index))
}

fn row_from_batch(
    batch: &RecordBatch,
    index: usize,
) -> Result<SearchRow, Box<dyn std::error::Error>> {
    let record_id = column_string(batch, "record_id", index)?;
    let path = column_string(batch, "path", index)?;
    let file_name = column_string(batch, "file_name", index)?;
    let extension = column_string(batch, "extension", index)?;
    let parent_dir = column_string(batch, "parent_dir", index)?;
    let modality = column_string(batch, "modality", index)?;
    let chunk = column_u32(batch, "chunk", index);
    let offset = column_u64(batch, "offset", index);
    let snippet = column_string(batch, "plaintext", index)?.replace(['\t', '\n'], " ");
    let unit = snippet_unit(&modality, &extension);
    let display_offset = display_offset(&extension, offset);
    Ok(SearchRow {
        record_id,
        path,
        file_name,
        extension,
        parent_dir,
        modality,
        chunk,
        offset,
        display_offset,
        unit,
        snippet,
        size_bytes: column_u64(batch, "size_bytes", index),
        modified_at: column_string(batch, "modified_at", index)?,
        distance: column_distance(batch, index),
    })
}

async fn search_rows(
    query_config: &QueryConfig,
    connection: &Connection,
    table_name: &str,
    query: &str,
    limit: usize,
    offset: usize,
) -> Result<Vec<SearchRow>, Box<dyn std::error::Error>> {
    let effective_limit = limit.saturating_add(offset);

    let table = open_table(connection, table_name)
        .await
        .ok_or("search table does not exist")?;
    let query_vector = read_query_embedding(query_config, query).await?;
    let mut results = table
        .vector_search(query_vector)?
        .limit(effective_limit)
        .execute()
        .await?;

    let mut global_index = 0usize;
    let mut rows = Vec::new();
    while let Some(batch) = results.next().await {
        let batch = batch?;
        for index in 0..batch.num_rows() {
            if global_index < offset {
                global_index += 1;
                continue;
            }
            if rows.len() >= limit {
                return Ok(rows);
            }
            rows.push(row_from_batch(&batch, index)?);
            global_index += 1;
        }
    }

    Ok(rows)
}

async fn run_search(
    args: &Args,
    connection: &Connection,
    table_target: &TableTarget,
    query: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let (range_start, range_end) = if let Some(range_value) = args.range.as_deref() {
        let (start, end) =
            parse_range(range_value).map_err(|err| format!("invalid --range: {err}"))?;
        (start, Some(end))
    } else {
        (0, None)
    };
    let effective_limit = range_end.map(|end| end - range_start).unwrap_or(args.limit);
    let rows = search_rows(
        &query_config_from_args(args),
        connection,
        &table_target.table_name,
        query,
        effective_limit,
        range_start,
    )
    .await?;

    if args.json {
        println!("{}", serde_json::to_string(&rows)?);
    } else {
        for row in rows {
            println!(
                "{}\t{}\t{}\t{}:{}\t{}",
                row.path, row.file_name, row.modality, row.unit, row.display_offset, row.snippet
            );
        }
    }

    Ok(())
}

async fn all_index_rows(
    connection: &Connection,
    table_name: &str,
) -> Result<Vec<SearchRow>, Box<dyn std::error::Error>> {
    let table = open_table(connection, table_name)
        .await
        .ok_or("search table does not exist")?;
    let mut results = table.query().limit(1_000_000).execute().await?;
    let mut rows = Vec::new();

    while let Some(batch) = results.next().await {
        let batch = batch?;
        for index in 0..batch.num_rows() {
            rows.push(row_from_batch(&batch, index)?);
        }
    }

    rows.sort_by(|left, right| {
        left.path
            .cmp(&right.path)
            .then(left.chunk.cmp(&right.chunk))
            .then(left.offset.cmp(&right.offset))
            .then(left.record_id.cmp(&right.record_id))
    });

    Ok(rows)
}

async fn all_text_rows(
    connection: &Connection,
    table_name: &str,
) -> Result<Vec<SearchRow>, Box<dyn std::error::Error>> {
    Ok(all_index_rows(connection, table_name)
        .await?
        .into_iter()
        .filter(|row| row.modality == "text" && !row.snippet.trim().is_empty())
        .collect())
}

fn merge_chunk_windows(windows: &mut Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    windows.sort_unstable();
    let mut merged: Vec<(u32, u32)> = Vec::new();

    for (start, end) in windows.drain(..) {
        if let Some((_, last_end)) = merged.last_mut() {
            if start <= last_end.saturating_add(1) {
                *last_end = (*last_end).max(end);
                continue;
            }
        }
        merged.push((start, end));
    }

    merged
}

fn build_match_context(rows: Vec<SearchRow>, term: &str, window: u32) -> MatchContextResponse {
    let term_lower = term.to_lowercase();
    let mut sources: BTreeMap<String, Vec<SearchRow>> = BTreeMap::new();

    for row in rows {
        sources.entry(row.path.clone()).or_default().push(row);
    }

    let mut context_sources = Vec::new();
    for (path, mut source_rows) in sources {
        source_rows.sort_by(|left, right| {
            left.chunk
                .cmp(&right.chunk)
                .then(left.offset.cmp(&right.offset))
                .then(left.record_id.cmp(&right.record_id))
        });

        let matching_chunks: BTreeSet<u32> = source_rows
            .iter()
            .filter(|row| row.snippet.to_lowercase().contains(&term_lower))
            .map(|row| row.chunk)
            .collect();

        if matching_chunks.is_empty() {
            continue;
        }

        let mut raw_windows: Vec<(u32, u32)> = matching_chunks
            .iter()
            .map(|chunk| (chunk.saturating_sub(window), chunk.saturating_add(window)))
            .collect();
        let merged_windows = merge_chunk_windows(&mut raw_windows);

        let windows = merged_windows
            .into_iter()
            .map(|(start_chunk, end_chunk)| {
                let chunks: Vec<SearchRow> = source_rows
                    .iter()
                    .filter(|row| row.chunk >= start_chunk && row.chunk <= end_chunk)
                    .cloned()
                    .collect();
                let text = chunks
                    .iter()
                    .map(|row| row.snippet.trim())
                    .filter(|snippet| !snippet.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n\n");
                MatchContextWindow {
                    start_chunk,
                    end_chunk,
                    text,
                    chunks,
                }
            })
            .collect();

        let first = &source_rows[0];
        context_sources.push(MatchContextSource {
            path,
            file_name: first.file_name.clone(),
            extension: first.extension.clone(),
            parent_dir: first.parent_dir.clone(),
            matches: matching_chunks.len(),
            windows,
        });
    }

    MatchContextResponse {
        term: term.to_string(),
        window,
        sources: context_sources,
    }
}

fn print_match_context(response: &MatchContextResponse) {
    println!(
        "# Match context for {:?} (window: {} chunk{})",
        response.term,
        response.window,
        if response.window == 1 { "" } else { "s" }
    );

    if response.sources.is_empty() {
        println!("\nNo matches.");
        return;
    }

    for source in &response.sources {
        println!("\n## {}", source.path);
        println!("matches: {}", source.matches);
        for window in &source.windows {
            println!(
                "\n### chunks {}-{}\n{}",
                window.start_chunk, window.end_chunk, window.text
            );
        }
    }
}

async fn run_match_context(
    args: &Args,
    connection: &Connection,
    table_target: &TableTarget,
    term: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let term = term.trim();
    if term.is_empty() {
        return Err("--match-context must not be empty".into());
    }

    let rows = all_text_rows(connection, &table_target.table_name).await?;
    let response = build_match_context(rows, term, args.context_window);

    if args.json {
        println!("{}", serde_json::to_string_pretty(&response)?);
    } else {
        print_match_context(&response);
    }

    Ok(())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn unix_mtime(metadata: &fs::Metadata) -> Option<u64> {
    metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs())
}

fn stable_fnv1a_hex(value: &str) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn safe_sidecar_name(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn document_type_from_extension(extension: &str) -> &'static str {
    match extension {
        ".pdf" | ".djvu" | ".epub" | ".ps" => "paged_document",
        ".doc" | ".docx" | ".odt" | ".rtf" | ".wp" | ".wp5" | ".wp6" | ".wpd" => "document",
        ".htm" | ".html" | ".xhtml" => "html_document",
        ".txt" | ".md" | ".rst" | ".msg" | ".art" | ".lst" => "text_document",
        ".zip" | ".tar" | ".tgz" | ".tbz2" | ".txz" | ".7z" | ".cbz" | ".cbr" => "archive",
        ".db3" | ".sqlite" | ".sqlite3" => "database",
        ".ipynb" => "notebook",
        _ => "unknown",
    }
}

fn title_guess_from_file_name(file_name: &str) -> Option<String> {
    let stem = Path::new(file_name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(file_name);
    let mut words = Vec::new();
    for raw_word in stem.split(|ch: char| {
        ch == '_' || ch == '-' || ch == '.' || ch == '+' || ch == '(' || ch == ')'
    }) {
        let word = raw_word.trim();
        if word.is_empty() {
            continue;
        }
        words.push(word.to_string());
    }
    if words.is_empty() {
        None
    } else {
        Some(words.join(" "))
    }
}

fn first_year(value: &str) -> Option<String> {
    let bytes = value.as_bytes();
    for index in 0..bytes.len().saturating_sub(3) {
        let window = &bytes[index..index + 4];
        if window.iter().all(|byte| byte.is_ascii_digit()) {
            let year = std::str::from_utf8(window).ok()?;
            if year.starts_with("18") || year.starts_with("19") || year.starts_with("20") {
                return Some(year.to_string());
            }
        }
    }
    None
}

fn all_years(value: &str) -> Vec<String> {
    let mut years = BTreeSet::new();
    let bytes = value.as_bytes();
    for index in 0..bytes.len().saturating_sub(3) {
        let window = &bytes[index..index + 4];
        if window.iter().all(|byte| byte.is_ascii_digit()) {
            if let Ok(year) = std::str::from_utf8(window) {
                if year.starts_with("18") || year.starts_with("19") || year.starts_with("20") {
                    years.insert(year.to_string());
                }
            }
        }
    }
    years.into_iter().collect()
}

fn citation_slug(title: &str) -> String {
    let mut slug = String::new();
    for ch in title.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
        } else if !slug.ends_with('-') {
            slug.push('-');
        }
        if slug.len() >= 48 {
            break;
        }
    }
    slug.trim_matches('-').to_string()
}

fn trim_candidate_token(value: &str) -> String {
    value
        .trim_matches(|ch: char| {
            ch.is_whitespace()
                || matches!(
                    ch,
                    ',' | ';' | ':' | '.' | ')' | '(' | ']' | '[' | '}' | '{' | '"' | '\''
                )
        })
        .to_string()
}

fn extract_doi_tokens(text: &str) -> Vec<String> {
    let mut dois = BTreeSet::new();
    for token in text.split_whitespace() {
        let normalized = token.trim_start_matches("doi:").trim_start_matches("DOI:");
        if let Some(index) = normalized.find("10.") {
            let candidate = trim_candidate_token(&normalized[index..]);
            if candidate.len() >= 7 && candidate.contains('/') {
                dois.insert(candidate);
            }
        }
    }
    dois.into_iter().collect()
}

fn extract_url_tokens(text: &str) -> Vec<String> {
    let mut urls = BTreeSet::new();
    for token in text.split_whitespace() {
        if token.starts_with("http://") || token.starts_with("https://") {
            let candidate = trim_candidate_token(token);
            if candidate.len() > "http://x".len() {
                urls.insert(candidate);
            }
        }
    }
    urls.into_iter().collect()
}

fn excerpt_chars(text: &str, max_chars: usize) -> String {
    let mut value: String = text.chars().take(max_chars).collect();
    if text.chars().count() > max_chars {
        value.push_str("...");
    }
    value
}

fn previous_char_boundary(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn citation_parenthetical_candidates(text: &str) -> Vec<String> {
    let mut candidates = BTreeSet::new();
    let bytes = text.as_bytes();
    for index in 0..bytes.len().saturating_sub(3) {
        let window = &bytes[index..index + 4];
        if !window.iter().all(|byte| byte.is_ascii_digit()) {
            continue;
        }
        let Ok(year) = std::str::from_utf8(window) else {
            continue;
        };
        if !(year.starts_with("18") || year.starts_with("19") || year.starts_with("20")) {
            continue;
        }
        let search_start = previous_char_boundary(text, index.saturating_sub(120));
        let Some(open_relative) = text[search_start..index].rfind('(') else {
            continue;
        };
        let open = search_start + open_relative;
        let search_end = previous_char_boundary(text, (index + 80).min(text.len()));
        let Some(close_relative) = text[index..search_end].find(')') else {
            continue;
        };
        let close = index + close_relative + 1;
        let candidate = text[open..close].trim();
        if candidate.len() >= 8
            && candidate.len() <= 220
            && candidate.chars().any(|ch| ch.is_ascii_alphabetic())
        {
            candidates.insert(candidate.to_string());
        }
    }
    candidates.into_iter().collect()
}

fn looks_like_reference_section(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("references")
        || lower.contains("bibliography")
        || lower.contains("literature cited")
        || lower.contains("works cited")
}

fn build_reference_candidate(
    row: &SearchRow,
    kind: &str,
    text: String,
    doi: Option<String>,
    url: Option<String>,
    year: Option<String>,
    confidence: f64,
    generated_unix: u64,
) -> ReferenceCandidate {
    let source_input = format!("{}\0{}\0{}", row.path, row.size_bytes, row.modified_at);
    let source_id = stable_fnv1a_hex(&source_input);
    let candidate_input = format!(
        "{}\0{}\0{}\0{}\0{}",
        row.path, row.chunk, row.offset, kind, text
    );
    ReferenceCandidate {
        schema_version: 1,
        candidate_id: stable_fnv1a_hex(&candidate_input),
        source_id,
        path: row.path.clone(),
        file_name: row.file_name.clone(),
        extension: row.extension.clone(),
        parent_dir: row.parent_dir.clone(),
        chunk: row.chunk,
        offset: row.offset,
        display_offset: row.display_offset,
        unit: row.unit,
        kind: kind.to_string(),
        text,
        doi,
        url,
        year,
        confidence,
        enrichment_run: EnrichmentRunMetadata {
            tool: "wolfe".to_string(),
            pass: "references-v1".to_string(),
            generated_unix,
        },
    }
}

fn reference_candidates_from_row(row: &SearchRow, generated_unix: u64) -> Vec<ReferenceCandidate> {
    if row.modality != "text" || row.snippet.trim().is_empty() {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    let years = all_years(&row.snippet);
    for doi in extract_doi_tokens(&row.snippet) {
        candidates.push(build_reference_candidate(
            row,
            "doi",
            doi.clone(),
            Some(doi),
            None,
            years.first().cloned(),
            0.9,
            generated_unix,
        ));
    }
    for url in extract_url_tokens(&row.snippet) {
        candidates.push(build_reference_candidate(
            row,
            "url",
            url.clone(),
            None,
            Some(url),
            years.first().cloned(),
            0.75,
            generated_unix,
        ));
    }
    for citation in citation_parenthetical_candidates(&row.snippet) {
        candidates.push(build_reference_candidate(
            row,
            "inline_citation",
            citation.clone(),
            None,
            None,
            first_year(&citation),
            0.55,
            generated_unix,
        ));
    }
    if looks_like_reference_section(&row.snippet) {
        candidates.push(build_reference_candidate(
            row,
            "reference_section",
            excerpt_chars(row.snippet.trim(), 1200),
            None,
            None,
            years.first().cloned(),
            0.45,
            generated_unix,
        ));
    }

    candidates
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut start = 0usize;
    for (index, ch) in text.char_indices() {
        if matches!(ch, '.' | '?' | '!') {
            let end = index + ch.len_utf8();
            let sentence = text[start..end].trim();
            if sentence.len() >= 20 {
                sentences.push(sentence.to_string());
            }
            start = end;
        }
    }
    let tail = text[start..].trim();
    if tail.len() >= 20 {
        sentences.push(tail.to_string());
    }
    sentences
}

fn normalized_phrase(value: &str) -> String {
    value
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn trim_concept_phrase(value: &str) -> String {
    let trimmed = value
        .trim_matches(|ch: char| {
            ch.is_whitespace()
                || matches!(
                    ch,
                    ',' | ';' | ':' | '.' | ')' | '(' | ']' | '[' | '}' | '{' | '"' | '\''
                )
        })
        .to_string();
    let mut words = trimmed.split_whitespace().collect::<Vec<_>>();
    while words.len() > 1
        && words[0].len() > 2
        && words[0].chars().all(|ch| ch.is_ascii_uppercase())
    {
        words.remove(0);
    }
    words.join(" ")
}

fn is_unhelpful_concept_phrase(value: &str) -> bool {
    let lower = normalized_phrase(value);
    if lower.len() < 3 {
        return true;
    }
    let words = lower.split_whitespace().collect::<Vec<_>>();
    if words.is_empty() || words.len() > 12 {
        return true;
    }
    matches!(
        words[0],
        "a" | "an"
            | "the"
            | "i"
            | "we"
            | "you"
            | "he"
            | "she"
            | "it"
            | "they"
            | "this"
            | "that"
            | "these"
            | "those"
            | "there"
            | "what"
            | "which"
            | "who"
            | "when"
            | "where"
            | "why"
            | "how"
    )
}

fn phrase_before_marker(sentence: &str, marker: &str) -> Option<String> {
    let index = sentence.find(marker)?;
    let before = sentence[..index].trim();
    let phrase = before
        .rsplit_once(|ch: char| matches!(ch, '.' | ';' | ':' | '(' | ')'))
        .map(|(_, right)| right)
        .unwrap_or(before);
    let phrase = trim_concept_phrase(phrase);
    let word_count = phrase.split_whitespace().count();
    if (1..=12).contains(&word_count)
        && phrase.chars().any(|ch| ch.is_alphabetic())
        && !is_unhelpful_concept_phrase(&phrase)
    {
        Some(phrase)
    } else {
        None
    }
}

fn phrase_after_marker(sentence: &str, marker: &str) -> Option<String> {
    let index = sentence.find(marker)?;
    let after = sentence[index + marker.len()..].trim();
    let mut phrase = after
        .split_once(|ch: char| matches!(ch, ',' | ';' | ':' | '(' | ')'))
        .map(|(left, _)| left)
        .unwrap_or(after);
    for delimiter in [" is ", " are ", " was ", " were ", " by ", " that "] {
        if let Some((left, _)) = phrase.split_once(delimiter) {
            phrase = left;
        }
    }
    let phrase = trim_concept_phrase(phrase);
    let word_count = phrase.split_whitespace().count();
    if (1..=12).contains(&word_count)
        && phrase.chars().any(|ch| ch.is_alphabetic())
        && !is_unhelpful_concept_phrase(&phrase)
    {
        Some(phrase)
    } else {
        None
    }
}

fn build_concept_candidate(
    row: &SearchRow,
    phrase: String,
    definition_type: &str,
    evidence_text: String,
    confidence: f64,
    generated_unix: u64,
) -> ConceptCandidate {
    let source_input = format!("{}\0{}\0{}", row.path, row.size_bytes, row.modified_at);
    let source_id = stable_fnv1a_hex(&source_input);
    let normalized = normalized_phrase(&phrase);
    let candidate_input = format!(
        "{}\0{}\0{}\0{}\0{}",
        row.path, row.chunk, row.offset, definition_type, normalized
    );
    ConceptCandidate {
        schema_version: 1,
        candidate_id: stable_fnv1a_hex(&candidate_input),
        source_id,
        path: row.path.clone(),
        file_name: row.file_name.clone(),
        extension: row.extension.clone(),
        parent_dir: row.parent_dir.clone(),
        chunk: row.chunk,
        offset: row.offset,
        display_offset: row.display_offset,
        unit: row.unit,
        phrase,
        normalized_phrase: normalized,
        definition_type: definition_type.to_string(),
        evidence_text,
        confidence,
        enrichment_run: EnrichmentRunMetadata {
            tool: "wolfe".to_string(),
            pass: "concepts-v1".to_string(),
            generated_unix,
        },
    }
}

fn concept_candidates_from_row(row: &SearchRow, generated_unix: u64) -> Vec<ConceptCandidate> {
    if row.modality != "text" || row.snippet.trim().is_empty() {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    for sentence in split_sentences(&row.snippet) {
        let lower = sentence.to_lowercase();
        let patterns = [
            (" is defined as ", "explicit_definition", 0.78),
            (" are defined as ", "explicit_definition", 0.78),
            (" means ", "denotation", 0.72),
            (" refers to ", "denotation", 0.72),
            (" is called ", "naming", 0.64),
            (" are called ", "naming", 0.64),
            (" is known as ", "naming", 0.64),
            (" are known as ", "naming", 0.64),
            (" is a ", "description", 0.58),
            (" is an ", "description", 0.58),
        ];
        for (marker, definition_type, confidence) in patterns {
            if !lower.contains(marker) {
                continue;
            }
            if let Some(phrase) = phrase_before_marker(&sentence, marker) {
                candidates.push(build_concept_candidate(
                    row,
                    phrase,
                    definition_type,
                    excerpt_chars(&sentence, 900),
                    confidence,
                    generated_unix,
                ));
            }
        }
        for marker in ["called ", "known as "] {
            if !lower.contains(marker) {
                continue;
            }
            if let Some(phrase) = phrase_after_marker(&sentence, marker) {
                candidates.push(build_concept_candidate(
                    row,
                    phrase,
                    "naming",
                    excerpt_chars(&sentence, 900),
                    0.5,
                    generated_unix,
                ));
            }
        }
    }

    candidates
}

fn build_bibtex_candidate(path: &str, file_name: &str, title: Option<&str>) -> BibtexCandidate {
    let year = first_year(path);
    let title_value = title.map(|value| value.to_string());
    let slug = title
        .map(citation_slug)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "source".to_string());
    let citation_key = match year.as_deref() {
        Some(year) => format!("unknown{year}-{slug}"),
        None => format!("unknown-{slug}"),
    };
    let bibtex_title = title_value
        .as_deref()
        .unwrap_or(file_name)
        .replace('{', "\\{")
        .replace('}', "\\}");
    let bibtex_year = year
        .as_ref()
        .map(|value| format!("  year = {{{value}}},\n"))
        .unwrap_or_default();
    let bibtex = format!(
        "@misc{{{citation_key},\n  title = {{{bibtex_title}}},\n{bibtex_year}  file = {{{path}}},\n  note = {{Wolfe filename-derived candidate; requires verification}}\n}}"
    );

    BibtexCandidate {
        entry_type: "misc".to_string(),
        citation_key,
        title: title_value,
        year,
        howpublished: path.to_string(),
        note: "filename-derived candidate; requires bibliographic verification".to_string(),
        confidence: 0.2,
        sources: vec!["file_name".to_string(), "path".to_string()],
        bibtex,
    }
}

fn collection_path(parent_dir: &str) -> Vec<String> {
    parent_dir
        .split(std::path::MAIN_SEPARATOR)
        .filter(|part| !part.is_empty())
        .map(|part| part.to_string())
        .collect()
}

fn word_count(text: &str) -> usize {
    text.split_whitespace()
        .filter(|word| word.chars().any(|ch| ch.is_alphanumeric()))
        .count()
}

fn build_source_metadata(
    path: String,
    mut rows: Vec<SearchRow>,
    generated_unix: u64,
) -> SourceMetadata {
    rows.sort_by(|left, right| {
        left.chunk
            .cmp(&right.chunk)
            .then(left.offset.cmp(&right.offset))
            .then(left.record_id.cmp(&right.record_id))
    });
    let first = &rows[0];
    let file_metadata = fs::metadata(&path).ok();
    let file_size_bytes = file_metadata.as_ref().map(|metadata| metadata.len());
    let file_modified_unix = file_metadata.as_ref().and_then(unix_mtime);
    let source_size_bytes = file_size_bytes.unwrap_or_else(|| {
        rows.iter()
            .map(|row| row.size_bytes)
            .max()
            .unwrap_or(first.size_bytes)
    });
    let mut modalities = BTreeMap::new();
    let mut unique_chunks = BTreeSet::new();
    let mut page_count = None;
    let mut text_char_count = 0usize;
    let mut total_word_count = 0usize;
    let mut text_chunk_count = 0usize;
    let mut image_chunk_count = 0usize;
    let mut other_chunk_count = 0usize;

    for row in &rows {
        *modalities.entry(row.modality.clone()).or_insert(0) += 1;
        unique_chunks.insert(row.chunk);
        if row.unit == "page" {
            page_count = Some(page_count.unwrap_or(0).max(row.display_offset));
        }
        match row.modality.as_str() {
            "text" => {
                text_chunk_count += 1;
                text_char_count += row.snippet.chars().count();
                total_word_count += word_count(&row.snippet);
            }
            "image" => image_chunk_count += 1,
            _ => other_chunk_count += 1,
        }
    }

    let title_guess = title_guess_from_file_name(&first.file_name);
    let bibtex_candidate = build_bibtex_candidate(&path, &first.file_name, title_guess.as_deref());
    let id_input = format!("{path}\0{source_size_bytes}\0{}", first.modified_at);
    SourceMetadata {
        schema_version: 1,
        source_id: stable_fnv1a_hex(&id_input),
        path,
        file_name: first.file_name.clone(),
        extension: first.extension.clone(),
        parent_dir: first.parent_dir.clone(),
        document_type: document_type_from_extension(&first.extension).to_string(),
        title_guess,
        collection_path: collection_path(&first.parent_dir),
        source_size_bytes,
        file_size_bytes,
        file_modified_unix,
        indexed_modified_at: first.modified_at.clone(),
        row_count: rows.len(),
        text_chunk_count,
        image_chunk_count,
        other_chunk_count,
        chunk_count: unique_chunks.len(),
        page_count,
        text_char_count,
        word_count: total_word_count,
        modalities,
        bibtex_candidate,
        enrichment_run: EnrichmentRunMetadata {
            tool: "wolfe".to_string(),
            pass: "source-metadata-v1".to_string(),
            generated_unix,
        },
    }
}

fn sidecar_path(metadata_root: &Path, source: &SourceMetadata) -> PathBuf {
    let prefix = source.source_id.get(0..2).unwrap_or("xx");
    metadata_root.join(prefix).join(format!(
        "{}-{}.wolfe-meta.json",
        source.source_id,
        safe_sidecar_name(&source.file_name)
    ))
}

async fn run_enrich_corpus(
    args: &Args,
    connection: &Connection,
    table_target: &TableTarget,
) -> Result<(), Box<dyn std::error::Error>> {
    let rows = all_index_rows(connection, &table_target.table_name).await?;
    let generated_unix = unix_now();
    let run_source_metadata =
        args.enrichment_pass == "source-metadata" || args.enrichment_pass == "all";
    let run_references = args.enrichment_pass == "references" || args.enrichment_pass == "all";
    let run_concepts = args.enrichment_pass == "concepts" || args.enrichment_pass == "all";
    if !run_source_metadata && !run_references && !run_concepts {
        return Err(
            "invalid --enrichment-pass: use source-metadata, references, concepts, or all".into(),
        );
    }

    let mut source_count = 0usize;
    let mut reference_count = 0usize;
    let mut concept_count = 0usize;

    if run_source_metadata {
        source_count = write_source_metadata(args, rows.clone(), generated_unix)?;
    }

    if run_references {
        reference_count = write_reference_candidates(args, &rows, generated_unix)?;
    }

    if run_concepts {
        concept_count = write_concept_candidates(args, &rows, generated_unix)?;
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&CorpusEnrichmentSummary {
            pass: args.enrichment_pass.clone(),
            sources: source_count,
            reference_candidates: reference_count,
            concept_candidates: concept_count,
            sidecar_root: args.metadata_root.to_string_lossy().to_string(),
            catalog: args.metadata_catalog.to_string_lossy().to_string(),
            reference_catalog: args.reference_catalog.to_string_lossy().to_string(),
            concept_catalog: args.concept_catalog.to_string_lossy().to_string(),
        })?
    );

    Ok(())
}

fn write_source_metadata(
    args: &Args,
    rows: Vec<SearchRow>,
    generated_unix: u64,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut grouped: BTreeMap<String, Vec<SearchRow>> = BTreeMap::new();
    for row in rows {
        grouped.entry(row.path.clone()).or_default().push(row);
    }

    fs::create_dir_all(&args.metadata_root)?;
    if let Some(parent) = args.metadata_catalog.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let catalog_file = fs::File::create(&args.metadata_catalog)?;
    let mut catalog = BufWriter::new(catalog_file);

    let mut source_count = 0;
    for (path, source_rows) in grouped {
        if source_rows.is_empty() {
            continue;
        }
        let metadata = build_source_metadata(path, source_rows, generated_unix);
        let target = sidecar_path(&args.metadata_root, &metadata);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        let sidecar = fs::File::create(target)?;
        serde_json::to_writer_pretty(sidecar, &metadata)?;
        serde_json::to_writer(&mut catalog, &metadata)?;
        writeln!(catalog)?;
        source_count += 1;
    }
    catalog.flush()?;

    Ok(source_count)
}

fn write_reference_candidates(
    args: &Args,
    rows: &[SearchRow],
    generated_unix: u64,
) -> Result<usize, Box<dyn std::error::Error>> {
    if let Some(parent) = args.reference_catalog.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let reference_file = fs::File::create(&args.reference_catalog)?;
    let mut writer = BufWriter::new(reference_file);
    let mut seen = BTreeSet::new();
    let mut count = 0usize;

    for row in rows {
        for candidate in reference_candidates_from_row(row, generated_unix) {
            if !seen.insert(candidate.candidate_id.clone()) {
                continue;
            }
            serde_json::to_writer(&mut writer, &candidate)?;
            writeln!(writer)?;
            count += 1;
        }
    }
    writer.flush()?;
    Ok(count)
}

fn write_concept_candidates(
    args: &Args,
    rows: &[SearchRow],
    generated_unix: u64,
) -> Result<usize, Box<dyn std::error::Error>> {
    if let Some(parent) = args.concept_catalog.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let concept_file = fs::File::create(&args.concept_catalog)?;
    let mut writer = BufWriter::new(concept_file);
    let mut seen = BTreeSet::new();
    let mut count = 0usize;

    for row in rows {
        for candidate in concept_candidates_from_row(row, generated_unix) {
            if !seen.insert(candidate.candidate_id.clone()) {
                continue;
            }
            serde_json::to_writer(&mut writer, &candidate)?;
            writeln!(writer)?;
            count += 1;
        }
    }
    writer.flush()?;
    Ok(count)
}

async fn connect_web_state(state: &WebState) -> Result<Connection, String> {
    connect(state.db_root.to_string_lossy().as_ref())
        .execute()
        .await
        .map_err(|err| err.to_string())
}

async fn web_search(
    State(state): State<WebState>,
    Query(params): Query<SearchParams>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let query = params.q.trim().to_string();
    if query.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "query must not be empty".to_string(),
        ));
    }
    let limit = params.limit.unwrap_or(20).clamp(1, 100);
    let offset = params.offset.unwrap_or(0);
    let connection = connect_web_state(&state)
        .await
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err))?;
    let results = search_rows(
        &state.query_config,
        &connection,
        &state.table_name,
        &query,
        limit,
        offset,
    )
    .await
    .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    Ok(Json(SearchResponse {
        query,
        limit,
        offset,
        results,
    }))
}

async fn context_rows(
    connection: &Connection,
    table_name: &str,
    params: &ContextParams,
) -> Result<Vec<SearchRow>, Box<dyn std::error::Error>> {
    let table = open_table(connection, table_name)
        .await
        .ok_or("search table does not exist")?;
    let window = params.window.unwrap_or(2).min(20);
    let path_filter = format!("path = {}", sql_quote(&params.path));
    let filter = if let Some(chunk) = params.chunk {
        let start = chunk.saturating_sub(window);
        let end = chunk.saturating_add(window);
        format!("{path_filter} AND chunk >= {start} AND chunk <= {end}")
    } else if let Some(offset) = params.offset {
        let span = u64::from(window).saturating_mul(2);
        let start = offset.saturating_sub(span);
        let end = offset.saturating_add(span);
        format!("{path_filter} AND offset >= {start} AND offset <= {end}")
    } else {
        path_filter
    };

    let mut results = table
        .query()
        .only_if(filter)
        .limit((window as usize * 2 + 1).max(20))
        .execute()
        .await?;
    let mut rows = Vec::new();
    while let Some(batch) = results.next().await {
        let batch = batch?;
        for index in 0..batch.num_rows() {
            rows.push(row_from_batch(&batch, index)?);
        }
    }
    rows.sort_by(|left, right| {
        left.chunk
            .cmp(&right.chunk)
            .then(left.offset.cmp(&right.offset))
            .then(left.record_id.cmp(&right.record_id))
    });
    Ok(rows)
}

async fn web_context(
    State(state): State<WebState>,
    Query(params): Query<ContextParams>,
) -> Result<Json<ContextResponse>, (StatusCode, String)> {
    if params.path.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "path must not be empty".to_string(),
        ));
    }
    let connection = connect_web_state(&state)
        .await
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err))?;
    let results = context_rows(&connection, &state.table_name, &params)
        .await
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    Ok(Json(ContextResponse {
        path: params.path,
        window: params.window.unwrap_or(2).min(20),
        results,
    }))
}

async fn web_index() -> impl IntoResponse {
    Html(WEB_INDEX)
}

async fn run_web(
    args: &Args,
    table_target: &TableTarget,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = WebState {
        db_root: table_target.db_root.clone(),
        table_name: table_target.table_name.clone(),
        query_config: query_config_from_args(args),
    };
    let app = Router::new()
        .route("/", get(web_index))
        .route("/api/search", get(web_search))
        .route("/api/context", get(web_context))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    eprintln!("wolfe web listening on http://{}", args.listen);
    axum::serve(listener, app).await?;
    Ok(())
}

const WEB_INDEX: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Wolfe Search</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #1d2528;
      --muted: #667176;
      --line: #d9dedb;
      --accent: #246b58;
      --accent-strong: #16483d;
      --warn: #8f4c21;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 15px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      border-bottom: 1px solid var(--line);
      background: #eef1ec;
      padding: 14px 20px;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 18px;
      padding: 18px;
      max-width: 1400px;
      margin: 0 auto;
    }
    form {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 92px 96px;
      gap: 8px;
      margin-bottom: 14px;
    }
    input, button {
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      background: var(--panel);
      color: var(--ink);
    }
    button {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
      cursor: pointer;
      font-weight: 600;
    }
    button.secondary {
      background: var(--panel);
      border-color: var(--line);
      color: var(--accent-strong);
      padding: 6px 9px;
      font-size: 13px;
    }
    button:disabled { opacity: .55; cursor: default; }
    .status {
      color: var(--muted);
      min-height: 22px;
      margin-bottom: 8px;
    }
    .results {
      display: grid;
      gap: 10px;
    }
    article, aside {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    article {
      padding: 12px;
    }
    .result-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 8px;
    }
    .file {
      font-weight: 650;
      word-break: break-word;
    }
    .path, .meta {
      color: var(--muted);
      font-size: 12px;
      word-break: break-all;
    }
    .snippet {
      margin: 8px 0;
      white-space: pre-wrap;
    }
    aside {
      padding: 12px;
      position: sticky;
      top: 18px;
      align-self: start;
      max-height: calc(100vh - 36px);
      overflow: auto;
    }
    aside h2 {
      font-size: 15px;
      margin: 0 0 10px;
    }
    .context-row {
      border-top: 1px solid var(--line);
      padding: 10px 0;
    }
    .context-row:first-of-type {
      border-top: 0;
    }
    .empty {
      color: var(--muted);
      padding: 14px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,.6);
    }
    .error { color: var(--warn); }
    @media (max-width: 880px) {
      main { grid-template-columns: 1fr; padding: 12px; }
      form { grid-template-columns: 1fr; }
      aside { position: static; max-height: none; }
    }
  </style>
</head>
<body>
  <header><h1>Wolfe Search</h1></header>
  <main>
    <section>
      <form id="search-form">
        <input id="query" name="q" autocomplete="off" placeholder="Search the Wolfe corpus" required>
        <input id="limit" name="limit" type="number" min="1" max="100" value="20" title="Limit">
        <button id="search-button" type="submit">Search</button>
      </form>
      <div id="status" class="status"></div>
      <div id="results" class="results"></div>
    </section>
    <aside>
      <h2>Context</h2>
      <div id="context" class="empty">Select a result to browse neighboring chunks.</div>
    </aside>
  </main>
  <script>
    const form = document.getElementById('search-form');
    const queryInput = document.getElementById('query');
    const limitInput = document.getElementById('limit');
    const button = document.getElementById('search-button');
    const statusBox = document.getElementById('status');
    const resultsBox = document.getElementById('results');
    const contextBox = document.getElementById('context');

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[ch]));
    }

    function meta(row) {
      const distance = row.distance == null ? '' : ` distance ${Number(row.distance).toFixed(4)}`;
      return `${row.modality} ${row.extension || ''} ${row.unit}:${row.display_offset} chunk ${row.chunk}${distance}`;
    }

    function renderResults(rows) {
      if (!rows.length) {
        resultsBox.innerHTML = '<div class="empty">No results.</div>';
        return;
      }
      resultsBox.innerHTML = rows.map((row, index) => `
        <article>
          <div class="result-head">
            <div>
              <div class="file">${escapeHtml(row.file_name || row.path)}</div>
              <div class="path">${escapeHtml(row.path)}</div>
              <div class="meta">${escapeHtml(meta(row))}</div>
            </div>
            <button class="secondary" type="button" data-index="${index}">Context</button>
          </div>
          <div class="snippet">${escapeHtml(row.snippet)}</div>
        </article>
      `).join('');
      resultsBox.querySelectorAll('button[data-index]').forEach((node) => {
        node.addEventListener('click', () => loadContext(rows[Number(node.dataset.index)]));
      });
    }

    function renderContext(payload) {
      if (!payload.results.length) {
        contextBox.className = 'empty';
        contextBox.textContent = 'No context rows found.';
        return;
      }
      contextBox.className = '';
      contextBox.innerHTML = payload.results.map((row) => `
        <div class="context-row">
          <div class="meta">${escapeHtml(meta(row))}</div>
          <div>${escapeHtml(row.snippet)}</div>
        </div>
      `).join('');
    }

    async function loadContext(row) {
      contextBox.className = 'empty';
      contextBox.textContent = 'Loading context...';
      const params = new URLSearchParams({
        path: row.path,
        chunk: row.chunk,
        window: '2'
      });
      const response = await fetch(`/api/context?${params}`);
      if (!response.ok) {
        contextBox.className = 'empty error';
        contextBox.textContent = await response.text();
        return;
      }
      renderContext(await response.json());
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const q = queryInput.value.trim();
      if (!q) return;
      button.disabled = true;
      statusBox.textContent = 'Searching...';
      resultsBox.innerHTML = '';
      const params = new URLSearchParams({ q, limit: limitInput.value || '20' });
      try {
        const response = await fetch(`/api/search?${params}`);
        if (!response.ok) throw new Error(await response.text());
        const payload = await response.json();
        statusBox.textContent = `${payload.results.length} results for "${payload.query}"`;
        renderResults(payload.results);
      } catch (err) {
        statusBox.innerHTML = `<span class="error">${escapeHtml(err.message || err)}</span>`;
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>
"#;

fn collect_files(
    path: &Path,
    ignore_matcher: &IgnoreMatcher,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    if should_ignore_path(path, ignore_matcher) {
        return Ok(Vec::new());
    }

    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }

    if !path.is_dir() {
        return Err(format!(
            "path does not exist or is not accessible: {}",
            path.display()
        )
        .into());
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
    if args.download_models {
        return download_models(&args);
    }
    let table_target = resolve_table_target(&args.db);
    let ignore_matcher = load_ignore_matcher(&args)?;
    let mode = match (
        args.web,
        args.path.as_deref(),
        args.path_list.as_deref(),
        args.search.as_deref(),
        args.match_context.as_deref(),
        args.enrich_corpus,
    ) {
        (true, None, None, None, None, false) => Mode::Web,
        (false, Some(path), None, None, None, false) => Mode::Ingest(path),
        (false, None, Some(path_list), None, None, false) => Mode::IngestList(path_list),
        (false, None, None, Some(query), None, false) => Mode::Search(query),
        (false, None, None, None, Some(term), false) => Mode::MatchContext(term),
        (false, None, None, None, None, true) => Mode::EnrichCorpus,
        (true, _, _, _, _, _) => return Err("--web cannot be combined with other modes".into()),
        (false, None, None, None, None, false) => {
            return Err(
                "either --path, --path-list, --search, --match-context, --enrich-corpus, or --web is required"
                    .into(),
            );
        }
        _ => {
            return Err(
                "use exactly one of --path, --path-list, --search, --match-context, --enrich-corpus, or --web"
                    .into(),
            );
        }
    };

    if matches!(mode, Mode::Ingest(_) | Mode::IngestList(_)) {
        fs::create_dir_all(&table_target.db_root)?;
    }

    if matches!(mode, Mode::Web) {
        return run_web(&args, &table_target).await;
    }

    let connection = connect(table_target.db_root.to_string_lossy().as_ref())
        .execute()
        .await?;

    if let Mode::Search(query) = mode {
        return run_search(&args, &connection, &table_target, query).await;
    }

    if let Mode::MatchContext(term) = mode {
        return run_match_context(&args, &connection, &table_target, term).await;
    }

    if matches!(mode, Mode::EnrichCorpus) {
        return run_enrich_corpus(&args, &connection, &table_target).await;
    }

    if let Mode::IngestList(path_list) = mode {
        return run_ingest_list(&args, &connection, &table_target, path_list).await;
    }

    let root_path = match mode {
        Mode::Ingest(path) => path,
        Mode::IngestList(_) => unreachable!(),
        Mode::Search(_) => unreachable!(),
        Mode::MatchContext(_) => unreachable!(),
        Mode::EnrichCorpus => unreachable!(),
        Mode::Web => unreachable!(),
    };

    if args.watch {
        run_watch(
            &args,
            &connection,
            &table_target,
            root_path,
            &ignore_matcher,
        )
        .await
    } else {
        run_ingest(
            &args,
            &connection,
            &table_target,
            root_path,
            &ignore_matcher,
        )
        .await
    }
}

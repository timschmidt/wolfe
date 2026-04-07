use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to a file or directory to embed recursively
    #[arg(short, long, value_name = "PATH", alias = "file")]
    path: PathBuf,

    /// Path to the local model directory
    #[arg(long, default_value = "jina-embeddings-v4")]
    model_dir: PathBuf,

    /// Embedding task name
    #[arg(long, default_value = "retrieval")]
    task: String,

    /// Path to the Python interpreter
    #[arg(long, default_value = "python3")]
    python: String,

    /// Execution device: auto, cpu, cuda, or mps
    #[arg(long, default_value = "auto")]
    device: String,

    /// Path to the embedding helper script
    #[arg(long, default_value = "scripts/embed.py")]
    script: PathBuf,
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let files = collect_files(&args.path)?;

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
        .stderr(Stdio::piped())
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

    let output = child.wait_with_output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("embedding script failed: {stderr}").into());
    }

    print!("{}", String::from_utf8_lossy(&output.stdout));
    eprint!("{}", String::from_utf8_lossy(&output.stderr));

    Ok(())
}

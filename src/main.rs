use std::path::PathBuf;
use std::process::Command;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to the input file
    #[arg(short, long, value_name = "FILE")]
    file: PathBuf,

    /// Path to the local model directory
    #[arg(long, default_value = "jina-embeddings-v4")]
    model_dir: PathBuf,

    /// Embedding task name
    #[arg(long, default_value = "retrieval")]
    task: String,

    /// Path to the Python interpreter
    #[arg(long, default_value = "python3")]
    python: String,

    /// Path to the embedding helper script
    #[arg(long, default_value = "scripts/embed.py")]
    script: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let output = Command::new(&args.python)
        .arg(&args.script)
        .arg("--file")
        .arg(&args.file)
        .arg("--model-dir")
        .arg(&args.model_dir)
        .arg("--task")
        .arg(&args.task)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("embedding script failed: {stderr}").into());
    }

    print!("{}", String::from_utf8_lossy(&output.stdout));

    Ok(())
}

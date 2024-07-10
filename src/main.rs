use anyhow::Context;
use clap::Parser;

use mdav::io::{read_csv, write_csv, CsvData};
use mdav::mdav::mdav;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input file
    #[arg(short, long)]
    input: String,
    /// Output file
    #[arg(short, long)]
    output: String,
    /// Minimum number of samples in every cluster
    #[arg(short, long)]
    k: usize,
    /// Delimiter for input/output files
    #[arg(short, long, default_value = ",")]
    delimiter: char,
    /// Columns to ignore (comma-separated)
    #[arg(long, value_delimiter = ',', num_args = 0..)]
    ignore_cols: Option<Vec<String>>,
}

fn main() {
    let args = Args::parse();
    let ignore_cols = args.ignore_cols.unwrap_or(vec![]);
    let records = read_csv(&args.input, args.delimiter, &ignore_cols)
        .context("Could not read input file")
        .unwrap();
    let anonymized_records = mdav(records.data, args.k);
    let data = CsvData {
        header: records.header.clone(),
        data: anonymized_records,
    };
    write_csv(&args.output, &data, args.delimiter)
        .context("Could not write output file")
        .unwrap();
}

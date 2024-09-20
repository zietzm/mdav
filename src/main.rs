use std::ops::{AddAssign, DivAssign};
use std::str::FromStr;

use anyhow::Context;
use clap::Parser;

use mdav::io::{read_csv, write_csv, CsvData};
use mdav::mdav::mdav;
use num::Float;

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
    /// Precision to use for floating point numbers
    #[arg(long, default_value = "64")]
    precision: usize,
}

fn main() {
    let args = Args::parse();
    match args.precision {
        32 => run::<f32>(args),
        64 => run::<f64>(args),
        _ => panic!("Invalid precision"),
    }
}

fn run<T: Float + FromStr + ToString + AddAssign + DivAssign + Send + Sync>(args: Args) {
    let ignore_cols = args.ignore_cols.unwrap_or_default();
    let records = read_csv::<T>(&args.input, args.delimiter, &ignore_cols)
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

use anyhow::Context;
use clap::Parser;

use mdav::io::{read_csv, write_csv, CsvData};
use mdav::mdav::mdav;
use mdav::mdav::FloatType;

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
    /// Include just centroids and counts or all anonymized records
    #[arg(long, default_value = "false")]
    just_centroids: bool,
}

fn main() {
    let args = Args::parse();
    match args.precision {
        32 => run::<f32>(args),
        64 => run::<f64>(args),
        _ => panic!("Invalid precision"),
    }
}

fn run<T: FloatType>(args: Args) {
    let ignore_cols = args.ignore_cols.unwrap_or_default();
    let mut records = read_csv::<T>(&args.input, args.delimiter, &ignore_cols)
        .context("Could not read input file")
        .unwrap();
    let mut anonymized_records = mdav(records.data, args.k)
        .context("Could not compute MDAV")
        .unwrap();
    let data = if args.just_centroids {
        anonymized_records
            .centroids
            .iter_mut()
            .zip(anonymized_records.n_occurrences.iter())
            .for_each(|(centroid, n_occurrences)| {
                centroid.push(T::from(*n_occurrences).unwrap());
            });
        records.header.push("n_occurrences".to_string());
        CsvData {
            header: records.header,
            data: anonymized_records.centroids,
        }
    } else {
        let expanded = anonymized_records.expand();
        CsvData {
            header: records.header,
            data: expanded,
        }
    };
    write_csv(&args.output, &data, args.delimiter)
        .context("Could not write output file")
        .unwrap();
}

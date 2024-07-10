use std::io::{BufRead, Write};

use anyhow::{anyhow, Result};

pub struct CsvData {
    pub header: Vec<String>,
    pub data: Vec<Vec<f32>>,
}

pub fn read_csv(filename: &str, delimiter: char, ignore_cols: &[String]) -> Result<CsvData> {
    let mut records = vec![];
    let mut header = vec![];
    let mut cols_to_ignore = vec![];

    let file = std::fs::File::open(filename)?;
    let reader = std::io::BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        if header.is_empty() {
            header = line
                .split(delimiter)
                .map(|value| value.to_string())
                .collect();
            cols_to_ignore = header
                .iter()
                .enumerate()
                .filter_map(|(i, value)| {
                    if ignore_cols.contains(value) {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();
            continue;
        }
        let mut values = Vec::new();
        for (i, value) in line.split(delimiter).enumerate() {
            if cols_to_ignore.contains(&i) {
                continue;
            }
            if let Ok(value) = value.parse::<f32>() {
                values.push(value);
            } else {
                return Err(anyhow!("Could not parse value: {}", value));
            }
        }
        records.push(values);
    }
    Ok(CsvData {
        header,
        data: records,
    })
}

pub fn write_csv(filename: &str, data: &CsvData, delimiter: char) -> Result<()> {
    let mut file = std::fs::File::create(filename)?;
    let header_line = data.header.join(&delimiter.to_string());
    file.write_all(header_line.as_bytes())?;
    file.write_all("\n".as_bytes())?;
    for record in &data.data {
        let values = record
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<String>>();
        let line = values.join(&delimiter.to_string());
        file.write_all(line.as_bytes())?;
        file.write_all("\n".as_bytes())?;
    }
    Ok(())
}

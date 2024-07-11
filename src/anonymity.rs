use std::collections::HashMap;
use std::ops::{AddAssign, DivAssign};

use num::Float;

// Compute the k-anonymity of a set of records.
// k-anonymity is the minimum number of times any record has been seen.
pub fn compute_k_anonymity<T: Float + AddAssign + DivAssign + Into<f64>>(
    records: &[Vec<T>],
) -> usize {
    let mut seen_records = HashMap::new();
    for record in records {
        let record_hash = hash_record(record);
        seen_records
            .entry(record_hash)
            .and_modify(|count| *count += 1_usize)
            .or_insert(1_usize);
    }
    *(seen_records.values().min().unwrap())
}

// Hash a record (Vec<f32> or Vec<f64>) by xoring its values.
fn hash_record<T: Float + AddAssign + DivAssign + Into<f64>>(record: &[T]) -> u64 {
    let mut hash = 0;
    for &value in record {
        hash ^= value.into().to_bits();
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_record() {
        let record = vec![1.0, 2.0, 3.0];
        let expected: u64 = 1.0_f64.to_bits() ^ 2.0_f64.to_bits() ^ 3.0_f64.to_bits();
        assert_eq!(hash_record(&record), expected);
    }

    #[test]
    fn test_compute_k_anonymity_1() {
        let records = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![10.0, 11.0, 12.0],
            vec![10.1, 11.1, 12.1],
        ];
        let expected: usize = 1;
        assert_eq!(compute_k_anonymity(&records), expected);
    }

    #[test]
    fn test_compute_k_anonymity_2() {
        let records = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![10.0, 11.0, 12.0],
            vec![10.0, 11.0, 12.0],
            vec![10.0, 11.0, 12.0],
        ];
        let expected: usize = 2;
        assert_eq!(compute_k_anonymity(&records), expected);
    }
}

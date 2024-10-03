use anyhow::{bail, Result};
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use log::info;
use num::{Float, NumCast};
use rayon::prelude::*;
use std::{
    fmt::{Debug, Display, Write},
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
    str::FromStr,
    sync::{Arc, Mutex},
};

pub trait FloatType =
    Float + AddAssign + DivAssign + MulAssign + SubAssign + Send + Sync + Display + FromStr + Debug;

#[derive(Clone)]
pub struct MdavResult<T: FloatType> {
    pub centroids: Vec<Vec<T>>,
    pub n_occurrences: Vec<usize>,
}

impl<T: FloatType> MdavResult<T> {
    pub fn new(centroids: Vec<Vec<T>>, n_occurrences: Vec<usize>) -> Result<Self> {
        if centroids.len() != n_occurrences.len() {
            bail!(
                "centroids.len() ({}) != n_occurrences.len() ({})",
                centroids.len(),
                n_occurrences.len()
            );
        }
        let result = Self {
            centroids,
            n_occurrences,
        };
        Ok(result)
    }

    /// Expand the so that each k-anonymized record appears k times
    pub fn expand(&self) -> Vec<Vec<T>> {
        let n_samples: usize = self.n_occurrences.iter().sum();
        let n_features: usize = self.centroids[0].len();
        let mut expanded = vec![vec![T::zero(); n_features]; n_samples];
        for (i, centroid) in self.centroids.iter().enumerate() {
            for (j, value) in centroid.iter().enumerate() {
                expanded[i][j] = *value;
            }
        }
        expanded
    }

    /// Write the occurrences to the end of each centroid
    pub fn collapse(self) -> Vec<Vec<T>> {
        let mut centroids = self.centroids;
        centroids
            .iter_mut()
            .zip(self.n_occurrences.iter())
            .for_each(|(centroid, &n)| {
                let new_value = T::from(n).unwrap();
                centroid.push(new_value);
            });
        centroids
    }
}

// Compute the MDAV-anonymized representation of a set of records.
// Records are represented as a vector of vectors of floats.
// k is the minimum number of samples in every cluster.
pub fn mdav<T: FloatType>(records: Vec<Vec<T>>, k: usize) -> Result<MdavResult<T>> {
    let assignments = assign_mdav(&records, k)?;
    let min_assignment = assignments.iter().min().unwrap();
    let max_assignment = assignments.iter().max().unwrap();
    info!("MDAV assignments: {}-{}", min_assignment, max_assignment);
    let n_clusters = assignments.iter().max().unwrap() + 1;
    let centroids = compute_centroids(&records, &assignments, n_clusters as usize);
    let mut n_occurrences = vec![0; n_clusters as usize];
    for assignment in assignments.iter() {
        n_occurrences[assignment - 1] += 1;
    }
    let result = MdavResult::new(centroids, n_occurrences)?;
    Ok(result)
}

// Compute the MDAV assignments for a given set of records.
// The records are represented as a vector of vectors of floats.
// k is the minimum number of samples in every cluster.
pub fn assign_mdav<T: FloatType>(records: &[Vec<T>], k: usize) -> Result<Vec<usize>> {
    // Pseudocode:
    // function mdav(records, k)
    // output assignments for records
    // while 2k points or more in the dataset remain to be assigned, do
    //     find the centroid of those remaining points
    //     find the furthest point P from the centroid
    //     find the furthest point Q from P
    //     select and group the k - 1 nearest points to P, along with P itself
    //     select and group the k - 1 nearest points to Q, along with Q itself
    //     remove the two groups just formed from the remaining points
    // if there are between k and 2k - 1 points remaining, do
    //     form a new group with the remaining points
    //  else
    //     compute the centroid of each group
    //     add each remaining point to its nearest group centroid
    let mut assignments = vec![0; records.len()];
    let mut n_remaining = records.len();
    let mut group_num = 0;
    let n_iterations = (records.len() as u64).div_ceil(2 * k as u64);
    let progress = ProgressBar::new(n_iterations);
    progress.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos:>7}/{len:7} ({eta})",
        )
        .unwrap()
        .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
            write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
        })
        .progress_chars("#>-"),
    );
    let mut centroid = compute_centroid(records, &assignments)?;
    let mut denominator = T::from(n_remaining).unwrap();
    while n_remaining >= 2 * k {
        group_num += 1;
        let p = find_furthest_point(records, &assignments, &centroid);
        let p_group_idx = k_nearest(k, &p, records, &assignments);
        update_assignments(&mut assignments, &p_group_idx, group_num);
        update_centroid(&mut centroid, &mut denominator, records, &p_group_idx);

        group_num += 1;
        let q = find_furthest_point(records, &assignments, &p);
        let q_group_idx = k_nearest(k, &q, records, &assignments);
        update_assignments(&mut assignments, &q_group_idx, group_num);
        update_centroid(&mut centroid, &mut denominator, records, &q_group_idx);
        n_remaining -= q_group_idx.len() + p_group_idx.len();
        progress.inc(1);
    }
    progress.finish_with_message("Finished MDAV");
    assert!(n_remaining < 2 * k, "Too many points remaining");
    let remaining_idx = get_remaining_idx(&assignments);
    assert_eq!(remaining_idx.len(), n_remaining);
    if n_remaining >= k {
        group_num += 1;
        update_assignments(&mut assignments, &remaining_idx, group_num);
    } else {
        let centroids = compute_centroids(records, &assignments, group_num);
        assign_to_nearest_centroid(records, &centroids, &mut assignments);
    }
    // Start assignments at 0
    assignments = assignments
        .iter()
        .map(|&assignment| assignment - 1)
        .collect();
    Ok(assignments)
}

// Compute the centroid among records that have not been assigned.
fn compute_centroid<T: FloatType>(records: &[Vec<T>], assignments: &[usize]) -> Result<Vec<T>> {
    if records.len() != assignments.len() {
        bail!(
            "records.len() ({}) != assignments.len() ({})",
            records.len(),
            assignments.len()
        );
    }
    let n_unassigned = Arc::new(Mutex::new(NumCast::from(0.0).unwrap()));
    let result = records
        .par_iter()
        .zip(assignments.par_iter())
        .filter_map(|(record, &assignment)| if assignment == 0 { Some(record) } else { None })
        .fold(
            || vec![T::zero(); records[0].len()],
            |mut centroid, record| {
                for (i, value) in record.iter().enumerate() {
                    centroid[i] += *value;
                }
                let mut n = n_unassigned.lock().unwrap();
                *n += NumCast::from(1.0).unwrap();
                centroid
            },
        )
        .reduce(
            || vec![T::zero(); records[0].len()],
            |a, b| a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect(),
        )
        .iter()
        .map(|value| *value / *n_unassigned.lock().unwrap())
        .collect();
    Ok(result)
}

/// Update the centroid by subtracting the points in the group.
fn update_centroid<T: FloatType>(
    centroid: &mut [T],
    denominator: &mut T,
    records: &[Vec<T>],
    group_idx: &[usize],
) {
    centroid
        .iter_mut()
        .for_each(|x| *x *= T::from(*denominator).unwrap());
    for idx in group_idx {
        centroid
            .iter_mut()
            .zip(records[*idx].iter())
            .for_each(|(centroid_value, record_value)| {
                *centroid_value -= *record_value;
            });
    }
    *denominator -= T::from(group_idx.len()).unwrap();
    centroid.iter_mut().for_each(|x| *x /= *denominator);
}

// Find the furthest point from a given centroid among those records that have not been assigned.
fn find_furthest_point<T: FloatType>(
    records: &[Vec<T>],
    assignments: &[usize],
    centroid: &[T],
) -> Vec<T> {
    let furthest_point = Arc::new(Mutex::new(vec![T::zero(); centroid.len()]));
    let max_distance = Arc::new(Mutex::new(T::zero()));
    records
        .par_iter()
        .zip(assignments.par_iter())
        .filter_map(|(record, &assignment)| if assignment == 0 { Some(record) } else { None })
        .for_each(|record| {
            let distance = distance(record, centroid);
            if distance > *max_distance.lock().unwrap() {
                max_distance.lock().unwrap().clone_from(&distance);
                furthest_point.lock().unwrap().clone_from(record);
            }
        });
    let result = furthest_point.lock().unwrap().to_vec();
    result
}

// Compute the Euclidean distance between two vectors.
fn distance<T: FloatType>(a: &[T], b: &[T]) -> T {
    let mut distance: T = NumCast::from(0.0).unwrap();
    for (i, value) in a.iter().enumerate() {
        distance += (*value - b[i]).powi(2);
    }
    distance.sqrt()
}

// Find the k nearest points to a given point among those records that have not been assigned.
fn k_nearest<T: FloatType>(
    k: usize,
    point: &[T],
    records: &[Vec<T>],
    assignments: &[usize],
) -> Vec<usize> {
    let mut distances: Vec<(T, usize)> = records
        .par_iter()
        .zip(assignments.par_iter())
        .enumerate()
        .filter_map(|(i, (record, &assignment))| {
            if assignment == 0 {
                Some((distance(record, point), i))
            } else {
                None
            }
        })
        .collect();
    distances.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    distances.iter().take(k).map(|d| d.1).collect()
}

// Update the assignments for a group of records.
fn update_assignments(assignments: &mut [usize], group_idx: &[usize], group_num: usize) {
    for idx in group_idx {
        assignments[*idx] = group_num;
    }
}

// Get the indices of records that have not been assigned.
fn get_remaining_idx(assignments: &[usize]) -> Vec<usize> {
    assignments
        .iter()
        .enumerate()
        .filter_map(|(i, &assignment)| if assignment == 0 { Some(i) } else { None })
        .collect()
}

// Compute the centroids for each cluster.
fn compute_centroids<T: FloatType>(
    records: &[Vec<T>],
    assignments: &[usize],
    n_clusters: usize,
) -> Vec<Vec<T>> {
    let mut centroids = vec![vec![T::zero(); records[0].len()]; n_clusters];
    let mut n_per_cluster = vec![T::zero(); n_clusters];
    for (i, record) in records.iter().enumerate() {
        let assignment = assignments[i];
        if assignment != 0 {
            n_per_cluster[assignment - 1] += NumCast::from(1.0).unwrap();
            for (j, record_value) in record.iter().enumerate() {
                centroids[assignment - 1][j] += *record_value;
            }
        }
    }
    centroids
        .iter_mut()
        .zip(n_per_cluster.iter())
        .for_each(|(centroid, &n)| {
            for centroid_value in centroid.iter_mut() {
                *centroid_value /= n;
            }
        });
    centroids
}

// Assign every remaining record to its nearest centroid.
fn assign_to_nearest_centroid<T: FloatType>(
    records: &[Vec<T>],
    centroids: &[Vec<T>],
    assignments: &mut [usize],
) {
    for (i, record) in records.iter().enumerate() {
        if assignments[i] != 0 {
            continue;
        }
        let mut min_distance = Float::max_value();
        let mut nearest_centroid_idx = 0;
        for (j, centroid) in centroids.iter().enumerate() {
            let distance = distance(record, centroid);
            if distance < min_distance {
                min_distance = distance;
                nearest_centroid_idx = j;
            }
        }
        assignments[i] = nearest_centroid_idx + 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx_eq<T: FloatType>(left: &[T], right: &[T], tol: T) {
        assert_eq!(left.len(), right.len());
        for (a, b) in left.iter().zip(right.iter()) {
            assert!((*a - *b).abs() <= tol, "{:?} != {:?}", left, right);
        }
    }

    #[test]
    fn test_compute_centroid_all() {
        let records = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![10.0, 11.0, 12.0],
            vec![10.1, 11.1, 12.1],
        ];
        let expected = [22.2 / 4.0, 26.2 / 4.0, 30.2 / 4.0];
        let result = compute_centroid(&records, &[0, 0, 0, 0]).unwrap();
        assert_approx_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn test_compute_centroid_some() {
        let records = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![10.0, 11.0, 12.0],
            vec![10.1, 11.1, 12.1],
        ];
        let assignments = vec![0, 0, 0, 1];
        let expected = vec![12.1 / 3.0, 15.1 / 3.0, 18.1 / 3.0];
        assert_eq!(compute_centroid(&records, &assignments).unwrap(), expected);
    }

    #[test]
    fn test_find_furthest_point() {
        let records = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![10.0, 11.0, 12.0],
            vec![10.1, 11.1, 12.1],
        ];
        let assignments = vec![0, 0, 0, 1];
        let centroid = compute_centroid(&records, &assignments).unwrap();
        let expected = vec![10.0, 11.0, 12.0];
        assert_eq!(
            find_furthest_point(&records, &assignments, &centroid),
            expected
        );
    }

    #[test]
    fn test_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(distance(&a, &b), 27.0_f32.sqrt());
    }

    #[test]
    fn test_k_nearest() {
        let records = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![10.0, 11.0, 12.0],
        ];
        let assignments = vec![0, 0, 0, 0];
        let k = 2;
        let expected = vec![0, 1];
        assert_eq!(k_nearest(k, &records[0], &records, &assignments), expected);
    }

    #[test]
    fn test_update_assignments() {
        let mut assignments = vec![0, 0, 0, 0];
        let group_idx = vec![0, 1, 2];
        let group_num = 1;
        update_assignments(&mut assignments, &group_idx, group_num);
        assert_eq!(assignments, vec![1, 1, 1, 0]);
    }

    #[test]
    fn test_assign_to_nearest_centroid() {
        let records = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![10.0, 11.0, 12.0],
            vec![10.1, 11.1, 12.1],
            vec![10.2, 11.2, 12.2],
        ];
        let centroids = vec![vec![10.15, 11.15, 12.15], vec![1.05, 2.05, 3.05]];
        let mut assignments = vec![2, 2, 0, 1, 1];
        assign_to_nearest_centroid(&records, &centroids, &mut assignments);
        assert_eq!(assignments, vec![2, 2, 1, 1, 1]);
    }

    #[test]
    fn test_assign_mdav_1() {
        let records = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![10.0, 11.0, 12.0],
            vec![10.1, 11.1, 12.1],
        ];
        let result = assign_mdav(&records, 2).unwrap();
        assert_eq!(result[0], result[1]); // First cluster
        assert_eq!(result[2], result[3]); // Second cluster
    }

    #[test]
    fn test_assign_mdav_2() {
        let records = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![10.0, 11.0, 12.0],
            vec![10.1, 11.1, 12.1],
            vec![10.2, 11.2, 12.2],
        ];
        let result = assign_mdav(&records, 2).unwrap();
        assert_eq!(result[0], result[1], "{:?}", result); // First cluster
        assert_eq!(result[2], result[3], "{:?}", result); // Second cluster
        assert_eq!(result[2], result[4], "{:?}", result); // Second cluster
    }

    #[test]
    fn test_assign_mdav_3() {
        let records = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![1.1, 2.1, 3.1],
            vec![10.2, 11.2, 12.2],
        ];
        let result = assign_mdav(&records, 3).unwrap();
        let expected = vec![0, 0, 0, 0];
        assert_eq!(result, expected, "{:?}", result);
    }

    #[test]
    fn test_assign_mdav_4() {
        let records = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![1.1, 2.1, 3.1],
            vec![10.2, 11.2, 12.2],
            vec![10.3, 11.3, 12.3],
        ];
        let result = assign_mdav(&records, 3).unwrap();
        let expected = vec![0, 0, 0, 0, 0];
        assert_eq!(result, expected, "{:?}", result);
    }

    #[test]
    fn test_assign_mdav_5() {
        let records = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![1.1, 2.1, 3.1],
            vec![10.2, 11.2, 12.2],
            vec![10.3, 11.3, 12.3],
            vec![10.4, 11.4, 12.4],
        ];
        let result = assign_mdav(&records, 3).unwrap();
        assert_eq!(result[0], result[1], "{:?}", result);
        assert_eq!(result[0], result[2], "{:?}", result);
        assert_eq!(result[3], result[4], "{:?}", result);
        assert_eq!(result[3], result[5], "{:?}", result);
        assert_ne!(result[0], result[3], "{:?}", result);
    }

    #[test]
    fn test_assign_mdav_6() {
        let records = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![1.1, 2.1, 3.1],
            vec![10.0, 11.0, 12.0],
            vec![10.1, 11.1, 12.1],
            vec![10.1, 11.1, 12.1],
            vec![10.1, 11.1, 12.1],
            vec![10.1, 11.1, 12.3],
            vec![10.1, 11.1, 12.3],
            vec![10.1, 16.1, 12.1],
            vec![10.1, 15.1, 12.1],
            vec![10.5, 11.5, 12.4],
            vec![10.1, 11.1, 14.1],
            vec![10.1, 11.1, 14.1],
            vec![10.1, 11.1, 12.1],
            vec![10.1, 11.1, 12.1],
            vec![11.1, 12.1, 13.1],
            vec![11.1, 12.1, 13.1],
        ];
        let result = assign_mdav(&records, 2).unwrap();
        assert_eq!(result[0], result[1], "{:?}", result); // First cluster
    }

    #[test]
    fn test_compute_centroid_2() {
        let records = vec![
            vec![10.1, 11.1, 12.1],
            vec![1.1, 2.1, 3.1],
            vec![1.1, 2.1, 3.1],
            vec![10.0, 11.0, 12.0],
        ];
        let expected = vec![5.575, 6.575, 7.575];
        let result = compute_centroid(&records, &[0, 0, 0, 0]).unwrap();
        assert_approx_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn test_update_centroid() {
        let vecs = vec![
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![2.0, 2.0],
        ];
        let mut assignments = vec![0, 0, 0, 0];
        let mut centroid = compute_centroid(&vecs, &assignments).unwrap();
        assert_eq!(centroid, vec![1.5, 1.5]);
        let group_idx = vec![0, 1];
        group_idx.iter().for_each(|&idx| {
            assignments[idx] = 1;
        });
        let mut denominator = NumCast::from(4.0).unwrap();
        update_centroid(&mut centroid, &mut denominator, &vecs, &group_idx);
        assert_eq!(centroid, vec![2.0, 2.0]);
        assert_eq!(denominator, NumCast::from(2.0).unwrap());
        let group_idx = vec![2];
        group_idx.iter().for_each(|&idx| {
            assignments[idx] = 1;
        });
        update_centroid(&mut centroid, &mut denominator, &vecs, &group_idx);
        assert_eq!(centroid, vec![2.0, 2.0]);
        assert_eq!(denominator, NumCast::from(1.0).unwrap());
    }
}

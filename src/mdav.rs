use std::ops::{AddAssign, DivAssign};

use num::{Float, NumCast};

// Compute the MDAV-anonymized representation of a set of records.
// Records are represented as a vector of vectors of floats.
// k is the minimum number of samples in every cluster.
pub fn mdav<T: Float + AddAssign + DivAssign>(records: Vec<Vec<T>>, k: usize) -> Vec<Vec<T>> {
    let assignments = assign_mdav(&records, k);
    let n_clusters = assignments.iter().max().unwrap() + 1;
    let centroids = compute_centroids(&records, &assignments, n_clusters as usize);
    assignments
        .iter()
        .map(|&assignment| centroids[assignment as usize - 1].clone())
        .collect()
}

// Compute the MDAV assignments for a given set of records.
// The records are represented as a vector of vectors of floats.
// k is the minimum number of samples in every cluster.
pub fn assign_mdav<T: Float + AddAssign + DivAssign>(records: &[Vec<T>], k: usize) -> Vec<u32> {
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
    let mut group_num = 1;
    while n_remaining >= 2 * k {
        let centroid = compute_centroid(records, &assignments);
        let p = find_furthest_point(records, &assignments, &centroid);
        let p_group_idx = k_nearest(k, &p, records, &assignments);
        update_assignments(&mut assignments, &p_group_idx, group_num);
        group_num += 1;

        let q = find_furthest_point(records, &assignments, &p);
        let q_group_idx = k_nearest(k, &q, records, &assignments);
        update_assignments(&mut assignments, &q_group_idx, group_num);
        group_num += 1;
        n_remaining -= p_group_idx.len() + q_group_idx.len();
    }
    assert!(n_remaining < 2 * k, "Too many points remaining");
    let remaining_idx = get_remaining_idx(&assignments);
    if n_remaining >= k {
        update_assignments(&mut assignments, &remaining_idx, group_num);
    } else {
        let centroids = compute_centroids(records, &assignments, group_num as usize);
        assign_to_nearest_centroid(records, &centroids, &mut assignments);
    }
    assignments
}

// Compute the centroid among records that have not been assigned.
fn compute_centroid<T: Float + AddAssign>(records: &[Vec<T>], assignments: &[u32]) -> Vec<T> {
    let mut n_unassigned: T = NumCast::from(0.0).unwrap();
    records
        .iter()
        .zip(assignments)
        .filter_map(|(record, &assignment)| if assignment == 0 { Some(record) } else { None })
        .fold(vec![T::zero(); records[0].len()], |mut centroid, record| {
            for (i, value) in record.iter().enumerate() {
                centroid[i] += *value;
            }
            n_unassigned += NumCast::from(1.0).unwrap();
            centroid
        })
        .iter()
        .map(|value| *value / n_unassigned)
        .collect()
}

// Find the furthest point from a given centroid among those records that have not been assigned.
fn find_furthest_point<T: Float + AddAssign + DivAssign + PartialOrd>(
    records: &[Vec<T>],
    assignments: &[u32],
    centroid: &[T],
) -> Vec<T> {
    let mut furthest_point = vec![T::zero(); centroid.len()];
    let mut max_distance = NumCast::from(0.0).unwrap();
    records
        .iter()
        .zip(assignments)
        .filter_map(|(record, &assignment)| if assignment == 0 { Some(record) } else { None })
        .for_each(|record| {
            let distance = distance(record, centroid);
            if distance > max_distance {
                max_distance = distance;
                furthest_point = record.clone();
            }
        });
    furthest_point
}

// Compute the Euclidean distance between two vectors.
fn distance<T: Float + PartialOrd + AddAssign>(a: &[T], b: &[T]) -> T {
    let mut distance: T = NumCast::from(0.0).unwrap();
    for (i, value) in a.iter().enumerate() {
        distance += (*value - b[i]).powi(2);
    }
    distance.sqrt()
}

// Find the k nearest points to a given point among those records that have not been assigned.
fn k_nearest<T: Float + AddAssign + DivAssign + PartialOrd>(
    k: usize,
    point: &[T],
    records: &[Vec<T>],
    assignments: &[u32],
) -> Vec<usize> {
    let mut distances: Vec<(T, usize)> = records
        .iter()
        .enumerate()
        .zip(assignments)
        .filter_map(|((i, record), &assignment)| {
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
fn update_assignments(assignments: &mut [u32], group_idx: &[usize], group_num: u32) {
    for idx in group_idx {
        assignments[*idx] = group_num;
    }
}

// Get the indices of records that have not been assigned.
fn get_remaining_idx(assignments: &[u32]) -> Vec<usize> {
    assignments
        .iter()
        .enumerate()
        .filter_map(|(i, &assignment)| if assignment == 0 { Some(i) } else { None })
        .collect()
}

// Compute the centroids for each cluster.
fn compute_centroids<T: Float + AddAssign + DivAssign>(
    records: &[Vec<T>],
    assignments: &[u32],
    n_clusters: usize,
) -> Vec<Vec<T>> {
    let mut centroids = vec![vec![T::zero(); records[0].len()]; n_clusters];
    let mut n_per_cluster = vec![T::zero(); n_clusters];
    for (i, record) in records.iter().enumerate() {
        let assignment = assignments[i] as usize;
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
fn assign_to_nearest_centroid<T: Float + AddAssign + DivAssign + PartialOrd>(
    records: &[Vec<T>],
    centroids: &[Vec<T>],
    assignments: &mut [u32],
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
        assignments[i] = nearest_centroid_idx as u32 + 1;
    }
}

#[cfg(test)]
mod tests {
    use num::Float;
    use std::ops::{AddAssign, DivAssign};

    use super::*;

    fn assert_approx_eq<T: Float + AddAssign + DivAssign + PartialOrd>(
        left: &[T],
        right: &[T],
        tol: T,
    ) {
        assert_eq!(left.len(), right.len());
        for (a, b) in left.iter().zip(right.iter()) {
            assert!((*a - *b).abs() <= tol);
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
        let result = compute_centroid(&records, &[0, 0, 0, 0]);
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
        assert_eq!(compute_centroid(&records, &assignments), expected);
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
        let centroid = compute_centroid(&records, &assignments);
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
        let result = assign_mdav(&records, 2);
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
        let result = assign_mdav(&records, 2);
        assert_eq!(result[0], result[1]); // First cluster
        assert_eq!(result[2], result[3]); // Second cluster
        assert_eq!(result[2], result[4]); // Second cluster
    }
}

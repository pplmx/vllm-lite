use crate::types::TokenId;

pub fn greedy_sample(logits: &[f32]) -> TokenId {
    logits
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(max_idx, max_val), (i, &val)| {
            if val > max_val {
                (i, val)
            } else {
                (max_idx, max_val)
            }
        })
        .0 as TokenId
}

pub fn sample_batch(logits_list: &[Vec<f32>], temperature: f32) -> Vec<TokenId> {
    logits_list
        .iter()
        .map(|logits| {
            // TODO: temperature sampling in Phase 2
            let _ = temperature;
            greedy_sample(logits)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_selects_max() {
        assert_eq!(greedy_sample(&[0.1, 0.5, 0.3]), 1);
    }

    #[test]
    fn test_greedy_first_on_tie() {
        assert_eq!(greedy_sample(&[0.5, 0.5, 0.3]), 0);
    }

    #[test]
    fn test_greedy_empty() {
        assert_eq!(greedy_sample(&[]), 0);
    }

    #[test]
    fn test_sample_batch() {
        let logits = vec![vec![0.1, 0.9], vec![0.8, 0.2]];
        assert_eq!(sample_batch(&logits, 0.0), vec![1, 0]);
    }

    #[test]
    fn test_greedy_all_same_logit() {
        assert_eq!(greedy_sample(&[0.5, 0.5, 0.5]), 0);
    }

    #[test]
    fn test_greedy_negative_logits() {
        assert_eq!(greedy_sample(&[-1.0, -0.5, 0.0]), 2);
    }

    #[test]
    fn test_greedy_large_vocab() {
        let mut logits = vec![0.0; 10000];
        logits[9999] = 1.0;
        assert_eq!(greedy_sample(&logits), 9999);
    }
}

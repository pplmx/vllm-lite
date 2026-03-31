use crate::types::TokenId;
use rand::thread_rng;
use rand::Rng;

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

pub fn temperature_sample(logits: &[f32], temperature: f32) -> TokenId {
    if temperature <= 0.0 || logits.is_empty() {
        return greedy_sample(logits);
    }

    let scaled: Vec<f32> = logits.iter().map(|x| x / temperature).collect();
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scaled.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let probs: Vec<f32> = exp.iter().map(|x| x / sum).collect();

    let mut rng = thread_rng();
    let r: f32 = rng.gen_range(0.0..1.0);
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return i as TokenId;
        }
    }
    (probs.len() - 1) as TokenId
}

pub fn top_p_sample(logits: &[f32], top_p: f32) -> TokenId {
    if top_p >= 1.0 || logits.is_empty() {
        return greedy_sample(logits);
    }

    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let max_val = indexed[0].1;
    let exp: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_val).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let mut probs: Vec<f32> = exp.iter().map(|x| x / sum).collect();

    let mut cumsum = 0.0;
    let mut cutoff = probs.len();
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum > top_p {
            cutoff = i + 1;
            break;
        }
    }

    probs.truncate(cutoff);
    let total: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= total);

    let mut rng = thread_rng();
    let r: f32 = rng.gen_range(0.0..1.0);
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return indexed[i].0 as TokenId;
        }
    }
    indexed[probs.len() - 1].0 as TokenId
}

pub fn sample_batch(logits_list: &[Vec<f32>], temperature: f32, top_p: f32) -> Vec<TokenId> {
    logits_list
        .iter()
        .map(|logits| {
            if top_p < 1.0 {
                top_p_sample(logits, top_p)
            } else if temperature > 0.0 {
                temperature_sample(logits, temperature)
            } else {
                greedy_sample(logits)
            }
        })
        .collect()
}

pub fn apply_repeat_penalty(logits: &mut [f32], seen_tokens: &[TokenId], penalty: f32) {
    if penalty == 1.0 || seen_tokens.is_empty() || logits.is_empty() {
        return;
    }

    let mut seen = std::collections::HashSet::new();
    for &token in seen_tokens {
        let idx = token as usize;
        if idx < logits.len() && seen.insert(token) {
            logits[idx] /= penalty;
        }
    }
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
        assert_eq!(sample_batch(&logits, 0.0, 1.0), vec![1, 0]);
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

    #[test]
    fn test_temperature_one_unchanged() {
        let logits = &[0.1, 0.5, 0.3];
        let result = temperature_sample(logits, 1.0);
        assert!(result < 3);
    }

    #[test]
    fn test_temperature_zero_reverts_to_greedy() {
        let logits = &[0.1, 0.9, 0.3];
        let result = temperature_sample(logits, 0.0);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_temperature_very_small() {
        let logits = &[0.1, 0.9, 0.3];
        let result = temperature_sample(logits, 0.01);
        assert!(result < 3);
    }

    #[test]
    fn test_temperature_very_large() {
        let logits = &[0.1, 0.9, 0.3];
        let _result = temperature_sample(logits, 10.0);
    }

    #[test]
    fn test_temperature_empty() {
        let result = temperature_sample(&[], 1.0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_top_p_one_equals_greedy() {
        let logits = &[0.1, 0.9, 0.3];
        let result = top_p_sample(logits, 1.0);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_top_p_small() {
        let logits = &[0.9, 0.05, 0.05];
        let result = top_p_sample(logits, 0.5);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_top_p_empty() {
        let result = top_p_sample(&[], 0.9);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_sample_batch_greedy() {
        let logits = vec![vec![0.1, 0.9], vec![0.8, 0.2], vec![0.3, 0.7]];
        let result = sample_batch(&logits, 0.0, 1.0);
        assert_eq!(result, vec![1, 0, 1]);
    }

    #[test]
    fn test_sample_batch_temperature_and_top_p() {
        let logits = vec![vec![0.1, 0.9], vec![0.5, 0.5]];
        let _result = sample_batch(&logits, 0.5, 0.8);
    }

    #[test]
    fn test_repeat_penalty_basic() {
        let mut logits = vec![0.5, 0.5, 0.5];
        let seen = vec![1];
        apply_repeat_penalty(&mut logits, &seen, 2.0);
        assert!(logits[1] < 0.5);
        assert_eq!(logits[0], 0.5);
        assert_eq!(logits[2], 0.5);
    }

    #[test]
    fn test_repeat_penalty_no_effect_at_one() {
        let mut logits = vec![0.5, 0.5];
        let seen = vec![0];
        apply_repeat_penalty(&mut logits, &seen, 1.0);
        assert_eq!(logits[0], 0.5);
        assert_eq!(logits[1], 0.5);
    }
}

pub type TokenId = u32;

pub struct Request {
    pub id: u64,
    pub prompt: Vec<TokenId>,
    pub max_tokens: usize,
}

mod engine;
mod scheduler;
mod kv_cache;
mod model;
mod types;
mod sampling;

use engine::engine::Engine;
use scheduler::scheduler::Scheduler;
use model::fake::FakeModel;
use types::request::Request;

fn main() {
    let mut scheduler = Scheduler::new();

    scheduler.add_request(Request {
        id: 1,
        prompt: vec![1, 2, 3],
        max_tokens: 50,
    });

    let model = FakeModel::new(32000);

    let mut engine = Engine {
        scheduler,
        model,
    };

    for _ in 0..10 {
        engine.step();
    }

    println!("Done");
}

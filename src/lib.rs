use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub fn setup(states: i32, actions: i32) {
    log(&format!("states: {}, actions: {}", states, actions,));
}

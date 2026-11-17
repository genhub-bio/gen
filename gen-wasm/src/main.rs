fn main() {
    let response = unsafe { gen_browser_actions::genhub_repository_command(std::ptr::null()) };
    unsafe { gen_browser_actions::genhub_free_string(response) };
}

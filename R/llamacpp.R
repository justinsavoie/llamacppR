#' Generate text using llama.cpp (CPU)
#'
#' @param model Path to a GGUF model file.
#' @param prompt Input prompt string.
#' @param n_predict Number of tokens to generate (default 32).
#' @return A single string containing prompt + generated continuation.
#' @export
llama_simple <- function(model, prompt = "Hello my name is", n_predict = 32L) {
  stopifnot(is.character(model), length(model) == 1L)
  stopifnot(is.character(prompt), length(prompt) == 1L)
  stopifnot(is.numeric(n_predict) || is.integer(n_predict))
  .Call(`_llamacppR_llama_simple`, as.character(model)[1], as.character(prompt)[1], as.integer(n_predict)[1])
}


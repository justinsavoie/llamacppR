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

#' Generate text with sampling controls (temperature/top-p/top-k)
#'
#' @param model Path to a GGUF model file.
#' @param prompt Input prompt string.
#' @param n_predict Number of tokens to generate (default 32).
#' @param temperature Softmax temperature (>0; 1.0 default). Set to 0 for greedy.
#' @param top_p Nucleus sampling probability (0-1; 1.0 disables).
#' @param top_k Keep top K tokens (0 disables).
#' @param min_p Minimum probability mass (0 disables).
#' @param repeat_penalty Repetition penalty (1.0 disables).
#' @param repeat_last_n Number of last tokens to penalize (e.g., 64; 0 disables).
#' @param presence_penalty Presence penalty (0 disables).
#' @param frequency_penalty Frequency penalty (0 disables).
#' @param seed RNG seed for stochastic sampling (-1 for default/randomized).
#' @return A single string containing prompt + generated continuation.
#' @export
llama_sampled <- function(
  model,
  prompt = "Hello my name is",
  n_predict = 32L,
  temperature = 1.0,
  top_p = 1.0,
  top_k = 0L,
  min_p = 0.0,
  repeat_penalty = 1.0,
  repeat_last_n = 64L,
  presence_penalty = 0.0,
  frequency_penalty = 0.0,
  seed = -1L
) {
  stopifnot(is.character(model), length(model) == 1L)
  stopifnot(is.character(prompt), length(prompt) == 1L)
  stopifnot(is.numeric(n_predict) || is.integer(n_predict))
  stopifnot(is.numeric(temperature), is.numeric(top_p), is.numeric(min_p))
  stopifnot(is.numeric(repeat_penalty), is.numeric(presence_penalty), is.numeric(frequency_penalty))
  stopifnot(is.numeric(top_k) || is.integer(top_k))
  stopifnot(is.numeric(repeat_last_n) || is.integer(repeat_last_n))
  stopifnot(is.numeric(seed) || is.integer(seed))

  .Call(`_llamacppR_llama_sampled`,
        as.character(model)[1],
        as.character(prompt)[1],
        as.integer(n_predict)[1],
        as.numeric(temperature)[1],
        as.numeric(top_p)[1],
        as.integer(top_k)[1],
        as.numeric(min_p)[1],
        as.numeric(repeat_penalty)[1],
        as.integer(repeat_last_n)[1],
        as.numeric(presence_penalty)[1],
        as.numeric(frequency_penalty)[1],
        as.integer(seed)[1]
  )
}

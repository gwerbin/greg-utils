#' Exit R
exit <- function() {
    quit("no")
}


#' head() for arbitrary-dimensional arrays
heada <- function(m, n = 6) {
    ndim <- length(dim(m))
    if (ndim == 1) {
        head(m, n)
    } else {
        do.call("[", replicate(ndim, 1:n, simplify = FALSE))
    }
}

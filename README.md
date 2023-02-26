# Re-implementation of Vishal Patil's _Lisp Interpreter In Rust_

[![CI](https://github.com/mo-xiaoming/a-lisp-rs/actions/workflows/build.yml/badge.svg)](https://github.com/mo-xiaoming/a-lisp-rs/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/mo-xiaoming/a-lisp-rs/branch/main/graph/badge.svg?token=04MMF2MJGH)](https://codecov.io/gh/mo-xiaoming/a-lisp-rs)

Just started reading this book and here is the example it promises to work

## One Example

```lisp
(
    (define fact
        (lambda (n)
            (if (< n 1)
                1
                (* n (fact (- n 1))))))
    (fact 5)
)

(
    (define pi 3.14)

    (define r 10)

    (define sqr
        (lambda (r) (* r r)))

    (define area
        (lambda (r) (* pi (sqr r))))

    (area r)
)

(
    (define odd
        (lambda (v) (= 1 (% v 2))))

    (define 1 (list 1 2 3 4 5))

    (reduce
        (lambda (x y)
            (or x y)) (map odd 1))
)

(
    (define add-n
        (lambda (n)
            (lambda (a) (+ n a))))

    (define add-3 (add-n 3))

    (define integers (list 1 2 3 4 5))

    (map add-3 integers)
)
```

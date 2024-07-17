;; optimal plan length: 36
(define (problem blocksworld-n13_r2024)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 - object)
    (:init (arm-empty) (on-table b1) (on b2 b7) (on b3 b1) (on b4 b5) (on b5 b2) (on b6 b11) (on b7 b13) (on-table b8) (on b9 b12) (on b10 b6) (on b11 b3) (on b12 b8) (on b13 b9) (clear b4) (clear b10))
    (:goal (and (on b1 b10) (on b2 b11) (on b3 b13) (on-table b4) (on b5 b3) (on b6 b5) (on-table b7) (on b8 b9) (on b9 b4) (on b10 b12) (on b11 b8) (on b12 b2) (on b13 b7) (clear b1) (clear b6)))
)

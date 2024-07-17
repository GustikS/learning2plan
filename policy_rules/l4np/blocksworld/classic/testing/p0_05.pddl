;; optimal plan length: 42
(define (problem blocksworld-n15_r2024)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 - object)
    (:init (arm-empty) (on-table b1) (on b2 b11) (on b3 b10) (on b4 b1) (on b5 b7) (on b6 b5) (on-table b7) (on b8 b15) (on b9 b6) (on b10 b2) (on b11 b14) (on b12 b8) (on b13 b4) (on b14 b9) (on b15 b13) (clear b3) (clear b12))
    (:goal (and (on b1 b13) (on b2 b3) (on b3 b14) (on-table b4) (on b5 b10) (on b6 b5) (on b7 b4) (on b8 b6) (on b9 b7) (on b10 b1) (on b11 b12) (on-table b12) (on b13 b15) (on b14 b11) (on b15 b2) (clear b8) (clear b9)))
)

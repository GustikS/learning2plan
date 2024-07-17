;; optimal plan length: 46
(define (problem blocksworld-n14_r2024)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 - object)
    (:init (arm-empty) (on-table b1) (on b2 b10) (on b3 b8) (on b4 b1) (on b5 b2) (on b6 b3) (on b7 b12) (on b8 b14) (on b9 b6) (on b10 b13) (on b11 b7) (on b12 b4) (on b13 b9) (on-table b14) (clear b5) (clear b11))
    (:goal (and (on b1 b13) (on b2 b3) (on b3 b14) (on-table b4) (on b5 b10) (on b6 b5) (on b7 b4) (on b8 b6) (on b9 b7) (on b10 b1) (on b11 b12) (on-table b12) (on b13 b2) (on b14 b11) (clear b8) (clear b9)))
)

;; optimal plan length: 34
(define (problem blocksworld-n12_r2024)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 - object)
    (:init (arm-empty) (on-table b1) (on-table b2) (on b3 b2) (on b4 b8) (on-table b5) (on b6 b10) (on b7 b12) (on b8 b11) (on-table b9) (on b10 b7) (on-table b11) (on b12 b4) (clear b1) (clear b3) (clear b5) (clear b6) (clear b9))
    (:goal (and (on b1 b7) (on b2 b11) (on b3 b9) (on b4 b3) (on b5 b6) (on b6 b8) (on b7 b10) (on-table b8) (on b9 b2) (on b10 b12) (on-table b11) (on b12 b5) (clear b1) (clear b4)))
)

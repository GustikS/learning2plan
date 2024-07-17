;; optimal plan length: 32
(define (problem blocksworld-n11_r2024)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 - object)
    (:init (arm-empty) (on-table b1) (on b2 b1) (on b3 b7) (on-table b4) (on b5 b9) (on b6 b11) (on b7 b10) (on-table b8) (on b9 b6) (on-table b10) (on b11 b3) (clear b2) (clear b4) (clear b5) (clear b8))
    (:goal (and (on b1 b9) (on b2 b7) (on b3 b11) (on b4 b6) (on b5 b4) (on b6 b8) (on-table b7) (on b8 b10) (on b9 b3) (on b10 b1) (on b11 b2) (clear b5)))
)

;; optimal plan length: 46
(define (problem blocksworld-n16_r2024)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 - object)
    (:init (arm-empty) (on-table b1) (on-table b2) (on b3 b9) (on b4 b11) (on b5 b1) (on b6 b12) (on b7 b6) (on b8 b14) (on b9 b16) (on b10 b7) (on b11 b3) (on b12 b15) (on b13 b8) (on b14 b5) (on-table b15) (on b16 b2) (clear b4) (clear b10) (clear b13))
    (:goal (and (on-table b1) (on b2 b15) (on b3 b4) (on b4 b16) (on-table b5) (on b6 b12) (on b7 b2) (on b8 b7) (on b9 b5) (on-table b10) (on b11 b9) (on b12 b8) (on b13 b14) (on b14 b10) (on b15 b3) (on b16 b13) (clear b1) (clear b6) (clear b11)))
)

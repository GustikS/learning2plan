;; optimal plan length: 62
(define (problem blocksworld-n22_r2024)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 - object)
    (:init (arm-empty) (on b1 b10) (on-table b2) (on b3 b13) (on b4 b5) (on b5 b15) (on b6 b9) (on b7 b2) (on b8 b16) (on b9 b11) (on b10 b6) (on b11 b14) (on b12 b20) (on b13 b22) (on b14 b3) (on b15 b17) (on b16 b21) (on b17 b1) (on b18 b8) (on b19 b12) (on b20 b7) (on-table b21) (on b22 b18) (clear b4) (clear b19))
    (:goal (and (on b1 b20) (on b2 b1) (on b3 b10) (on b4 b7) (on b5 b9) (on-table b6) (on-table b7) (on b8 b21) (on b9 b12) (on b10 b2) (on b11 b6) (on b12 b8) (on b13 b15) (on b14 b19) (on b15 b16) (on b16 b18) (on b17 b5) (on b18 b22) (on b19 b3) (on b20 b4) (on-table b21) (on b22 b14) (clear b11) (clear b13) (clear b17)))
)

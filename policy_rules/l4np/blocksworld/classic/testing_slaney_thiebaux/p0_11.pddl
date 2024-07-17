;; optimal plan length: 64
(define (problem blocksworld-n21_r2024)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 - object)
    (:init (arm-empty) (on b1 b9) (on-table b2) (on b3 b11) (on b4 b16) (on b5 b14) (on b6 b2) (on-table b7) (on b8 b10) (on b9 b3) (on b10 b13) (on b11 b15) (on b12 b21) (on b13 b1) (on b14 b4) (on b15 b17) (on b16 b20) (on b17 b7) (on b18 b12) (on b19 b6) (on b20 b8) (on b21 b19) (clear b5) (clear b18))
    (:goal (and (on b1 b19) (on b2 b1) (on b3 b6) (on-table b4) (on b5 b9) (on b6 b2) (on-table b7) (on b8 b20) (on b9 b11) (on b10 b15) (on b11 b7) (on b12 b8) (on-table b13) (on b14 b18) (on b15 b17) (on b16 b5) (on b17 b21) (on b18 b3) (on b19 b4) (on b20 b14) (on b21 b13) (clear b10) (clear b12) (clear b16)))
)

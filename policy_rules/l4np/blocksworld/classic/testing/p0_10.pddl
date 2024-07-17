;; optimal plan length: 56
(define (problem blocksworld-n20_r2024)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 - object)
    (:init (arm-empty) (on b1 b8) (on-table b2) (on b3 b5) (on b4 b13) (on b5 b7) (on b6 b2) (on b7 b9) (on-table b8) (on b9 b12) (on b10 b15) (on b11 b18) (on b12 b20) (on b13 b3) (on b14 b16) (on b15 b19) (on b16 b10) (on b17 b11) (on b18 b6) (on-table b19) (on-table b20) (clear b1) (clear b4) (clear b14) (clear b17))
    (:goal (and (on b1 b2) (on b2 b12) (on-table b3) (on b4 b7) (on b5 b17) (on b6 b14) (on b7 b20) (on b8 b3) (on b9 b13) (on b10 b19) (on b11 b10) (on b12 b16) (on b13 b4) (on b14 b5) (on b15 b9) (on b16 b6) (on-table b17) (on b18 b15) (on b19 b1) (on b20 b11) (clear b8) (clear b18)))
)

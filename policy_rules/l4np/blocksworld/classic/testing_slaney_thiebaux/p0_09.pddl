;; optimal plan length: 46
(define (problem blocksworld-n19_r2024)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 - object)
    (:init (arm-empty) (on b1 b7) (on-table b2) (on b3 b4) (on b4 b12) (on-table b5) (on b6 b2) (on b7 b5) (on b8 b13) (on b9 b11) (on b10 b17) (on b11 b19) (on b12 b14) (on b13 b15) (on b14 b18) (on b15 b9) (on b16 b10) (on b17 b6) (on b18 b8) (on-table b19) (clear b1) (clear b3) (clear b16))
    (:goal (and (on-table b1) (on b2 b9) (on b3 b6) (on-table b4) (on b5 b10) (on b6 b18) (on b7 b4) (on-table b8) (on b9 b17) (on b10 b15) (on b11 b12) (on b12 b3) (on b13 b8) (on-table b14) (on-table b15) (on b16 b13) (on b17 b14) (on b18 b19) (on b19 b7) (clear b1) (clear b2) (clear b5) (clear b11) (clear b16)))
)

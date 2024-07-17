;; optimal plan length: 40
(define (problem blocksworld-n18_r2024)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 - object)
    (:init (arm-empty) (on-table b1) (on-table b2) (on b3 b11) (on b4 b3) (on b5 b2) (on b6 b13) (on b7 b4) (on b8 b14) (on b9 b8) (on b10 b18) (on-table b11) (on b12 b9) (on b13 b17) (on-table b14) (on b15 b10) (on b16 b5) (on b17 b7) (on b18 b16) (clear b1) (clear b6) (clear b12) (clear b15))
    (:goal (and (on-table b1) (on b2 b9) (on b3 b6) (on-table b4) (on b5 b10) (on b6 b18) (on b7 b4) (on-table b8) (on b9 b17) (on b10 b15) (on b11 b12) (on b12 b3) (on b13 b8) (on-table b14) (on-table b15) (on b16 b13) (on b17 b14) (on b18 b7) (clear b1) (clear b2) (clear b5) (clear b11) (clear b16)))
)

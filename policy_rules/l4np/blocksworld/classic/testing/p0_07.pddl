;; optimal plan length: 42
(define (problem blocksworld-n17_r2024)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 - object)
    (:init (arm-empty) (on b1 b8) (on-table b2) (on b3 b10) (on b4 b6) (on b5 b2) (on b6 b3) (on b7 b13) (on b8 b7) (on b9 b15) (on b10 b17) (on-table b11) (on b12 b16) (on b13 b11) (on b14 b9) (on b15 b5) (on b16 b4) (on-table b17) (clear b1) (clear b12) (clear b14))
    (:goal (and (on b1 b12) (on b2 b17) (on-table b3) (on b4 b5) (on b5 b15) (on-table b6) (on b7 b14) (on b8 b9) (on b9 b2) (on-table b10) (on b11 b6) (on-table b12) (on b13 b11) (on b14 b10) (on b15 b16) (on b16 b3) (on b17 b4) (clear b1) (clear b7) (clear b8) (clear b13)))
)

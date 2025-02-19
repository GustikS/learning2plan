
(define (problem transport-17)
 (:domain transport)
 (:objects 
    v1 - vehicle
    p1 p2 p3 p4 p5 p6 p7 p8 p9 - package
    l1 l2 l3 l4 l5 l6 l7 l8 l9 l10 - location
    c0 c1 c2 - size
    )
 (:init
    (capacity v1 c2)
    (capacity-predecessor c0 c1)
    (capacity-predecessor c1 c2)
    (at p1 l5)
    (at p2 l1)
    (at p3 l7)
    (at p4 l6)
    (at p5 l6)
    (at p6 l6)
    (at p7 l7)
    (at p8 l7)
    (at p9 l5)
    (at v1 l2)
    (road l2 l4)
    (road l10 l8)
    (road l2 l7)
    (road l3 l1)
    (road l7 l10)
    (road l6 l1)
    (road l9 l4)
    (road l8 l10)
    (road l10 l7)
    (road l10 l1)
    (road l4 l2)
    (road l4 l9)
    (road l1 l10)
    (road l7 l2)
    (road l1 l6)
    (road l2 l5)
    (road l1 l3)
    (road l5 l2)
    (road l6 l7)
    (road l7 l6)
    (road l1 l7)
    (road l7 l1)
    (road l5 l7)
    (road l7 l5)
    (road l2 l9)
    (road l9 l2)
    (road l8 l9)
    (road l9 l8)
    (road l7 l8)
    (road l8 l7)
    (road l3 l4)
    (road l4 l3)
    )
 (:goal  (and 
    (at p1 l6)
    (at p2 l10)
    (at p3 l6)
    (at p4 l4)
    (at p5 l1)
    (at p6 l4)
    (at p7 l5)
    (at p8 l8)
    (at p9 l9))))


(define (problem transport-05)
 (:domain transport)
 (:objects 
    v1 - vehicle
    p1 p2 p3 - package
    l1 l2 l3 l4 l5 l6 - location
    c0 c1 c2 - size
    )
 (:init
    (capacity v1 c2)
    (capacity-predecessor c0 c1)
    (capacity-predecessor c1 c2)
    (at p1 l4)
    (at p2 l2)
    (at p3 l6)
    (at v1 l4)
    (road l6 l2)
    (road l3 l1)
    (road l6 l1)
    (road l5 l4)
    (road l4 l5)
    (road l2 l6)
    (road l5 l3)
    (road l1 l6)
    (road l1 l3)
    (road l3 l5)
    (road l5 l6)
    (road l6 l5)
    )
 (:goal  (and 
    (at p1 l2)
    (at p2 l5)
    (at p3 l1))))

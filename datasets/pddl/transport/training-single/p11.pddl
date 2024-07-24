
(define (problem transport-11)
 (:domain transport)
 (:objects 
    v1 - vehicle
    p1 p2 - package
    l1 l2 l3 l4 l5 l6 - location
    c0 c1 c2 - size
    )
 (:init
    (capacity v1 c2)
    (capacity-predecessor c0 c1)
    (capacity-predecessor c1 c2)
    (at p1 l3)
    (at p2 l4)
    (at v1 l1)
    (road l6 l2)
    (road l1 l5)
    (road l3 l1)
    (road l6 l1)
    (road l4 l1)
    (road l5 l1)
    (road l1 l4)
    (road l2 l6)
    (road l1 l6)
    (road l1 l3)
    (road l1 l2)
    (road l2 l1)
    )
 (:goal  (and 
    (at p1 l1)
    (at p2 l1))))

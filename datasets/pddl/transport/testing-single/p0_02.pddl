
(define (problem transport-02)
 (:domain transport)
 (:objects 
    v1 - vehicle
    p1 - package
    l1 l2 l3 l4 l5 - location
    c0 c1 c2 - size
    )
 (:init
    (capacity v1 c2)
    (capacity-predecessor c0 c1)
    (capacity-predecessor c1 c2)
    (at p1 l1)
    (at v1 l1)
    (road l3 l1)
    (road l5 l4)
    (road l2 l3)
    (road l4 l5)
    (road l3 l2)
    (road l2 l5)
    (road l1 l3)
    (road l5 l2)
    (road l2 l4)
    (road l4 l2)
    )
 (:goal  (and 
    (at p1 l2))))

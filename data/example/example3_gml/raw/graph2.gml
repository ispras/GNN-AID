graph [
  node [
    id 0
    label "1"
    a 15
    b "alpha"
  ]
  node [
    id 1
    label "2"
    a 25
    b "beta"
  ]
  node [
    id 2
    label "3"
    a 35
    b "gamma"
  ]
  node [
    id 3
    label "4"
    a 45
    b "delta"
  ]
  edge [
    source 0
    target 1
    weight 1.2
    type "mixed"
  ]
  edge [
    source 0
    target 3
    weight 4.5
    type "hybrid"
  ]
  edge [
    source 1
    target 2
    weight 2.3
    type "complex"
  ]
  edge [
    source 2
    target 3
    weight 3.4
  ]
]

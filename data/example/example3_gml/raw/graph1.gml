graph [
  node [
    id 0
    label "1"
    a 10
    b "alpha"
  ]
  node [
    id 1
    label "2"
    a 20
    b "beta"
  ]
  node [
    id 2
    label "3"
    a 30
    b "gamma"
  ]
  node [
    id 3
    label "4"
    a 40
    b "delta"
  ]
  edge [
    source 0
    target 1
    weight 1.5
    type "mixed"
  ]
  edge [
    source 0
    target 3
    weight 0.9
    type "hybrid"
  ]
  edge [
    source 1
    target 2
    weight 2.7
  ]
  edge [
    source 2
    target 3
    type "complex"
  ]
]

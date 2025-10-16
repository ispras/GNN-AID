graph [
  node [
    id 0
    label "1"
    a 20
    b "alpha"
  ]
  node [
    id 1
    label "2"
    a 30
    b "beta"
  ]
  node [
    id 2
    label "3"
    a 40
    b "gamma"
  ]
  node [
    id 3
    label "4"
    a 50
    b "delta"
  ]
  node [
    id 4
    label "5"
    a 60
  ]
  edge [
    source 0
    target 1
    weight 1.8
    type "mixed"
  ]
  edge [
    source 0
    target 4
    weight 3.2
  ]
  edge [
    source 1
    target 2
  ]
  edge [
    source 2
    target 3
    weight 2.5
    type "complex"
  ]
  edge [
    source 3
    target 4
    type "hybrid"
  ]
]

// Copyright 2023 The Chai Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
)

// ComplexMatrix is a complex matrix
type ComplexMatrix struct {
	Cols   int
	Rows   int
	Data   []complex128
	States [][]complex128
}

// NewComplexMatrix creates a new complex matrix
func NewComplexMatrix(states, cols, rows int) ComplexMatrix {
	m := ComplexMatrix{
		Cols: cols,
		Rows: rows,
		Data: make([]complex128, 0, cols*rows),
	}
	if states > 0 {
		m.States = make([][]complex128, states)
		for i := range m.States {
			m.States[i] = make([]complex128, cols*rows)
		}
	}
	return m
}

// NewRandComplexMatrix creates a new random complex matrix
func NewRandComplexMatrix(rnd *rand.Rand, states, cols, rows int) ComplexMatrix {
	m := ComplexMatrix{
		Cols: cols,
		Rows: rows,
		Data: make([]complex128, 0, cols*rows),
	}
	factor := math.Sqrt(2.0 / float64(cols))
	for i := 0; i < cols*rows; i++ {
		m.Data = append(m.Data, complex(rnd.NormFloat64()*factor, rnd.NormFloat64()*factor))
	}
	if states > 0 {
		m.States = make([][]complex128, states)
		for i := range m.States {
			m.States[i] = make([]complex128, cols*rows)
		}
	}
	return m
}

// Size is the size of the complex matrix
func (m ComplexMatrix) Size() int {
	return m.Cols * m.Rows
}

func complexDot(X, Y []complex128) complex128 {
	var sum complex128
	for i, x := range X {
		sum += x * Y[i]
	}
	return sum
}

// ComplexMul multiplies two complex matrices
func ComplexMul(m ComplexMatrix, n ComplexMatrix) ComplexMatrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := ComplexMatrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]complex128, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, complexDot(mm, nn))
		}
	}
	return o
}

// ComplexAdd adds two complex matrices
func ComplexAdd(m ComplexMatrix, n ComplexMatrix) ComplexMatrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := ComplexMatrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]complex128, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value+n.Data[i%lenb])
	}
	return o
}

// ComplexActivation is a complex activation function
func ComplexActivation(m ComplexMatrix) ComplexMatrix {
	o := ComplexMatrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]complex128, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		var v complex128
		if real(value) > 0 {
			v = 1
		} else {
			v = -1
		}
		if imag(value) > 0 {
			v += 1i
		} else {
			v += -1i
		}
		o.Data = append(o.Data, v)
	}
	return o
}

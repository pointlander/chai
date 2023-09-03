// Copyright 2023 The Chai Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
)

const (
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Matrix is a matrix
type Matrix struct {
	Cols   int
	Rows   int
	Data   []float64
	States [][]float64
}

// NewMatrix creates a new matrix
func NewMatrix(states, cols, rows int) Matrix {
	m := Matrix{
		Cols: cols,
		Rows: rows,
		Data: make([]float64, 0, cols*rows),
	}
	if states > 0 {
		m.States = make([][]float64, states)
		for i := range m.States {
			m.States[i] = make([]float64, cols*rows)
		}
	}
	return m
}

// NewRandMatrix creates a new random matrix
func NewRandMatrix(rnd *rand.Rand, states, cols, rows int) Matrix {
	m := Matrix{
		Cols: cols,
		Rows: rows,
		Data: make([]float64, 0, cols*rows),
	}
	factor := math.Sqrt(2.0 / float64(cols))
	for i := 0; i < cols*rows; i++ {
		m.Data = append(m.Data, rnd.NormFloat64()*factor)
	}
	if states > 0 {
		m.States = make([][]float64, states)
		for i := range m.States {
			m.States[i] = make([]float64, cols*rows)
		}
	}
	return m
}

// Size is the size of the matrix
func (m Matrix) Size() int {
	return m.Cols * m.Rows
}

// Mul multiplies two matrices
func Mul(m Matrix, n Matrix) Matrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]float64, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, dot(mm, nn))
		}
	}
	return o
}

// H element wise multiplies two matrices
func H(m Matrix, n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value*n.Data[i%lenb])
	}
	return o
}

// Add adds two matrices
func Add(m Matrix, n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value+n.Data[i%lenb])
	}
	return o
}

// Sub subtracts two matrices
func Sub(m Matrix, n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value-n.Data[i%lenb])
	}
	return o
}

// Softmax is the softmax of a matrix
func Softmax(m Matrix) Matrix {
	size, width := len(m.Data), m.Cols
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	max := 0.0
	for _, v := range m.Data {
		if v > max {
			max = v
		}
	}
	values := make([]float64, width)
	for i := 0; i < size; i += width {
		s := max * S
		sum := 0.0
		for j, ax := range m.Data[i : i+width] {
			values[j] = math.Exp(ax - s)
			sum += values[j]
		}
		for _, cx := range values {
			o.Data = append(o.Data, cx/sum)
		}
	}
	return o
}

// Normalize normalizes a matrix to the unit vector
func Normalize(m Matrix) Matrix {
	size, width := len(m.Data), m.Cols
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i := 0; i < size; i += width {
		sum := 0.0
		for _, ax := range m.Data[i : i+width] {
			sum += ax * ax
		}
		length := math.Sqrt(sum)
		if sum == 0 {
			length = 1
		}
		for _, ax := range m.Data[i : i+width] {
			o.Data = append(o.Data, ax/length)
		}
	}
	return o
}

// Entropy is the entropy of the matrix
func Entropy(m Matrix) Matrix {
	size, width := len(m.Data), m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: 1,
		Data: make([]float64, 0, m.Rows),
	}
	for i := 0; i < size; i += width {
		sum := 0.0
		for k := 0; k < width; k++ {
			ax := m.Data[i+k]
			sum += ax * math.Log(ax)
		}
		o.Data = append(o.Data, -sum)
	}
	return o
}

// Neg negates a matrix
func Neg(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, -value)
	}
	return o
}

// Sigmoid computes the sigmoid of a matrix
func Sigmoid(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, 1/(1+math.Exp(-value)))
	}
	return o
}

// T tramsposes a matrix
func T(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Rows,
		Rows: m.Cols,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i := 0; i < m.Cols; i++ {
		for j := 0; j < m.Rows; j++ {
			o.Data = append(o.Data, m.Data[j*m.Cols+i])
		}
	}
	return o
}

func softmax(values []float64) {
	max := 0.0
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := 0.0
	for j, value := range values {
		values[j] = math.Exp(value - s)
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

// SelfEntropy computes the self entropy of Q, K, V
func SelfEntropy(Q, K, V Matrix) []float64 {
	entropies, values, results := make([]float64, V.Cols), make([]float64, K.Rows), make([]float64, 0, K.Rows)
	V = T(V)
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			entropies[j] = dot(values, V)
		}
		softmax(entropies)

		entropy := 0.0
		for _, e := range entropies {
			entropy += e * math.Log(e)
		}
		results = append(results, entropy)
	}
	return results
}

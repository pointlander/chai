// Copyright 2023 The Chai Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"runtime"
	"sort"

	"github.com/pointlander/datum/iris"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

const (
	cols, rows, middle = 4, 4, 1
)

// Autoencoder is a neural network
type AutoencoderNetwork struct {
	Theta []float64
	W1    ComplexMatrix
	B1    ComplexMatrix
	W2    ComplexMatrix
	B2    ComplexMatrix
	W3    ComplexMatrix
	B3    ComplexMatrix
	W4    ComplexMatrix
	B4    ComplexMatrix
}

// Infer computes the output of the network
func (a AutoencoderNetwork) Infer(inputs ComplexMatrix) (output ComplexMatrix) {
	l1 := EverettActivation(ComplexAdd(ComplexMul(a.W1, inputs), a.B1))
	l2 := EverettActivation(ComplexAdd(ComplexMul(a.W2, l1), a.B2))
	l3 := EverettActivation(ComplexAdd(ComplexMul(a.W3, l2), a.B3))
	l4 := ComplexAdd(ComplexMul(a.W4, l3), a.B4)
	return l4
}

// Middle layer
func (a AutoencoderNetwork) Middle(inputs ComplexMatrix) (output ComplexMatrix) {
	l1 := EverettActivation(ComplexAdd(ComplexMul(a.W1, inputs), a.B1))
	l2 := ComplexAdd(ComplexMul(a.W2, l1), a.B2)
	return l2
}

// Test tests the autoencoder
func (a AutoencoderNetwork) Test(fisher []iris.Iris) {
	//correct, incorrect := 0, 0
	for k, value := range fisher {
		inputs := NewComplexMatrix(0, cols, 1)
		for i := 0; i < cols; i++ {
			inputs.Data = append(inputs.Data, 0)
		}
		for j := range inputs.Data {
			inputs.Data[j] = cmplx.Rect(1, value.Measures[j])
		}
		l2 := a.Middle(inputs)
		fmt.Printf("%d %s", k, value.Label)
		for _, v := range l2.Data {
			fmt.Printf(" %f", cmplx.Phase(v))
		}
		fmt.Println()
	}
	//fmt.Println("correct", correct, float64(correct)/float64(correct+incorrect))
	//fmt.Println("incorrect", incorrect, float64(incorrect)/float64(correct+incorrect))
}

// Distribution is a distribution
type Distribution struct {
	Mean   float64
	StdDev float64
}

// Genome is a genome
type Genome struct {
	Theta   []Distribution
	W1      []Distribution
	B1      []Distribution
	W2      []Distribution
	B2      []Distribution
	W3      []Distribution
	B3      []Distribution
	W4      []Distribution
	B4      []Distribution
	Fitness Stat
	Rank    float64
	Cached  bool
}

// NewGenome creates a new genome
func NewGenome(rng *rand.Rand) Genome {
	factor := math.Sqrt(2.0 / float64(cols))
	theta := make([]Distribution, 0, 4)
	for i := 0; i < 4; i++ {
		theta = append(theta, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
	}
	w1 := make([]Distribution, 0, 2*cols*rows)
	for i := 0; i < 2*cols*rows; i++ {
		w1 = append(w1, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
	}
	b1 := make([]Distribution, 0, 2*rows)
	for i := 0; i < 2*rows; i++ {
		b1 = append(b1, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
	}
	w2 := make([]Distribution, 0, 2*2*rows*middle)
	for i := 0; i < 2*2*rows*middle; i++ {
		w2 = append(w2, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
	}
	b2 := make([]Distribution, 0, 2*middle)
	for i := 0; i < 2*middle; i++ {
		b2 = append(b2, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
	}
	w3 := make([]Distribution, 0, 2*2*middle*cols)
	for i := 0; i < 2*2*middle*cols; i++ {
		w3 = append(w3, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
	}
	b3 := make([]Distribution, 0, 2*cols)
	for i := 0; i < 2*cols; i++ {
		b3 = append(b3, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
	}
	w4 := make([]Distribution, 0, 2*2*cols*cols)
	for i := 0; i < 2*2*cols*cols; i++ {
		w4 = append(w4, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
	}
	b4 := make([]Distribution, 0, 2*cols)
	for i := 0; i < 2*cols; i++ {
		b4 = append(b4, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
	}
	g := Genome{
		Theta: theta,
		W1:    w1,
		B1:    b1,
		W2:    w2,
		B2:    b2,
		W3:    w3,
		B3:    b3,
		W4:    w4,
		B4:    b4,
	}
	return g
}

// Copy copies a genome
func (g *Genome) Copy() Genome {
	theta := make([]Distribution, len(g.Theta))
	copy(theta, g.Theta)
	w1 := make([]Distribution, len(g.W1))
	copy(w1, g.W1)
	b1 := make([]Distribution, len(g.B1))
	copy(b1, g.B1)
	w2 := make([]Distribution, len(g.W2))
	copy(w2, g.W2)
	b2 := make([]Distribution, len(g.B2))
	copy(b2, g.B2)
	w3 := make([]Distribution, len(g.W3))
	copy(w3, g.W3)
	b3 := make([]Distribution, len(g.B3))
	copy(b3, g.B3)
	w4 := make([]Distribution, len(g.W4))
	copy(w4, g.W4)
	b4 := make([]Distribution, len(g.B4))
	copy(b4, g.B4)
	return Genome{
		Theta: theta,
		W1:    w1,
		B1:    b1,
		W2:    w2,
		B2:    b2,
		W3:    w3,
		B3:    b3,
		W4:    w4,
		B4:    b4,
	}
}

// SampleAutoencoder samples an autoencoder
func (g *Genome) SampleAutoencoder(rng *rand.Rand) (network AutoencoderNetwork) {
	n := AutoencoderNetwork{}
	n.Theta = make([]float64, 0, 4)
	for i := 0; i < 4; i++ {
		n.Theta = append(n.Theta, rng.NormFloat64()*g.Theta[i].StdDev+g.Theta[i].Mean)
	}
	n.W1 = NewComplexMatrix(0, cols, rows)
	for i := 0; i < len(g.W1); i += 2 {
		a := g.W1[i]
		b := g.W1[i+1]
		n.W1.Data = append(n.W1.Data, complex(rng.NormFloat64()*a.StdDev+a.Mean, rng.NormFloat64()*b.StdDev+b.Mean))
	}
	n.B1 = NewComplexMatrix(0, 1, rows)
	for i := 0; i < len(g.B1); i += 2 {
		x := g.B1[i]
		y := g.B1[i+1]
		n.B1.Data = append(n.B1.Data, complex(rng.NormFloat64()*x.StdDev+x.Mean, rng.NormFloat64()*y.StdDev+y.Mean))
	}
	n.W2 = NewComplexMatrix(0, 2*cols, middle)
	for i := 0; i < len(g.W2); i += 2 {
		a := g.W2[i]
		b := g.W2[i+1]
		n.W2.Data = append(n.W2.Data, complex(rng.NormFloat64()*a.StdDev+a.Mean, rng.NormFloat64()*b.StdDev+b.Mean))
	}
	n.B2 = NewComplexMatrix(0, 1, middle)
	for i := 0; i < len(g.B2); i += 2 {
		x := g.B2[i]
		y := g.B2[i+1]
		n.B2.Data = append(n.B2.Data, complex(rng.NormFloat64()*x.StdDev+x.Mean, rng.NormFloat64()*y.StdDev+y.Mean))
	}
	n.W3 = NewComplexMatrix(0, 2*middle, cols)
	for i := 0; i < len(g.W3); i += 2 {
		a := g.W3[i]
		b := g.W3[i+1]
		n.W3.Data = append(n.W3.Data, complex(rng.NormFloat64()*a.StdDev+a.Mean, rng.NormFloat64()*b.StdDev+b.Mean))
	}
	n.B3 = NewComplexMatrix(0, 1, cols)
	for i := 0; i < len(g.B3); i += 2 {
		x := g.B3[i]
		y := g.B3[i+1]
		n.B3.Data = append(n.B3.Data, complex(rng.NormFloat64()*x.StdDev+x.Mean, rng.NormFloat64()*y.StdDev+y.Mean))
	}
	n.W4 = NewComplexMatrix(0, 2*cols, rows)
	for i := 0; i < len(g.W4); i += 2 {
		a := g.W4[i]
		b := g.W4[i+1]
		n.W4.Data = append(n.W4.Data, complex(rng.NormFloat64()*a.StdDev+a.Mean, rng.NormFloat64()*b.StdDev+b.Mean))
	}
	n.B4 = NewComplexMatrix(0, 1, rows)
	for i := 0; i < len(g.B4); i += 2 {
		x := g.B4[i]
		y := g.B4[i+1]
		n.B4.Data = append(n.B4.Data, complex(rng.NormFloat64()*x.StdDev+x.Mean, rng.NormFloat64()*y.StdDev+y.Mean))
	}
	return n
}

// Autoencoder is a neural network trained on the iris dataset
func Autoencoder(seed int) {
	cpus := runtime.NumCPU()
	rng := rand.New(rand.NewSource(int64(seed)))
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	min, max := make([]float64, 4), make([]float64, 4)
	for i := range min {
		min[i] = math.MaxFloat64
	}
	for i := range datum.Fisher {
		for j, v := range datum.Fisher[i].Measures {
			if v < min[j] {
				min[j] = v
			}
			if v > max[j] {
				max[j] = v
			}
		}
	}
	for i := range datum.Fisher {
		for j, v := range datum.Fisher[i].Measures {
			datum.Fisher[i].Measures[j] = 2 * math.Pi * (v - min[j]) / (max[j] - min[j])
		}
	}
	target := make([][][]float64, 3)
	const points = 50
	target[0] = [][]float64{}
	for i := 0; i < points; i++ {
		target[0] = append(target[0], datum.Fisher[i].Measures)
	}
	target[1] = [][]float64{}
	for i := 0; i < points; i++ {
		target[1] = append(target[1], datum.Fisher[i+50].Measures)
	}
	target[2] = [][]float64{}
	for i := 0; i < points; i++ {
		target[2] = append(target[2], datum.Fisher[i+100].Measures)
	}
	fmt.Println(datum.Fisher[0].Label, datum.Fisher[1].Label)
	fmt.Println(datum.Fisher[50].Label, datum.Fisher[51].Label)
	fmt.Println(datum.Fisher[100].Label, datum.Fisher[101].Label)

	const pop = 256
	pool := make([]Genome, 0, pop)
	for i := 0; i < pop; i++ {
		pool = append(pool, NewGenome(rng))
	}

	sample := func(rng *rand.Rand, g *Genome) (samples plotter.Values, stats []Stat, network *AutoencoderNetwork, found bool) {
		stats = make([]Stat, 1)
		scale := 128
		for i := 0; i < scale; i++ {
			n := g.SampleAutoencoder(rng)
			inputs := NewComplexMatrix(0, cols, 1)
			for i := 0; i < cols; i++ {
				inputs.Data = append(inputs.Data, 0)
			}
			fitness := 0.0
			phases := NewMatrix(0, middle, points*len(target))
			for _, class := range target {
				for _, v := range class {
					for j := range inputs.Data {
						inputs.Data[j] = cmplx.Rect(1, v[j])
					}
					l2 := n.Infer(inputs)
					for j, vv := range l2.Data {
						x := v[j]
						if x > math.Pi {
							x -= 2 * math.Pi
						}
						fit := cmplx.Phase(vv) - x
						fitness += fit * fit
					}
					middle := n.Middle(inputs)
					for _, vv := range middle.Data {
						phases.Data = append(phases.Data, cmplx.Phase(vv))
					}
				}
			}
			entropy := SelfEntropy(phases, phases, phases)
			sum := 0.0
			for _, v := range entropy {
				sum += v
			}
			sum = -sum / float64(len(entropy))
			fitness /= float64(points * len(target))
			fitness += sum
			samples = append(samples, fitness)
			stats[0].Add(float64(fitness))
			if fitness <= 7.5 {
				fmt.Println(i, fitness)
				network = &n
				found = true
				break
			}
		}

		for i := range stats {
			stats[i].Normalize()
		}
		return samples, stats, network, found
	}

	done := false
	d := make(plotter.Values, 0, 8)
	for i := range pool {
		dd, stats, network, found := sample(rng, &pool[i])
		fmt.Println(i, stats[0].Mean, stats[0].StdDev)
		if found {
			network.Test(datum.Fisher)
			done = true
			break
		}
		pool[i].Fitness = stats[0]
		pool[i].Cached = true
		d = append(d, dd...)
	}

	p := plot.New()
	p.Title.Text = "autoencoder"

	histogram, err := plotter.NewHist(d, 10)
	if err != nil {
		panic(err)
	}
	p.Add(histogram)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "autoencoder.png")
	if err != nil {
		panic(err)
	}

	rngs, generation := make(map[int]*rand.Rand), 0
Search:
	for !done {
		/*graph := pagerank.NewGraph64()
		for i := range pool {
			for j := i + 1; j < len(pool); j++ {
				// http://homework.uoregon.edu/pub/class/es202/ztest.html
				avga := pool[i].Fitness.Mean
				avgb := pool[j].Fitness.Mean
				avg := avga - avgb
				if avg < 0 {
					avg = -avg
				}
				stddeva := pool[i].Fitness.StdDev
				stddevb := pool[j].Fitness.StdDev
				stddev := math.Sqrt(stddeva*stddeva + stddevb*stddevb)
				z := stddev / avg
				graph.Link(uint64(i), uint64(j), z)
				graph.Link(uint64(j), uint64(i), z)
			}
		}
		graph.Rank(0.85, 0.000001, func(node uint64, rank float64) {
			pool[node].Rank = rank
		})*/
		sort.Slice(pool, func(i, j int) bool {
			return pool[i].Fitness.Mean < pool[j].Fitness.Mean
			//return pool[i].Rank > pool[j].Rank
		})
		pool = pool[:pop]
		if generation > 50 {
			network := pool[0].SampleAutoencoder(rng)
			network.Test(datum.Fisher)
			break
		}
		fmt.Println(generation, pool[0].Fitness.Mean, pool[0].Fitness.StdDev)
		if pool[0].Fitness.Mean < 1e-32 {
			break Search
		}
		for i := 0; i < pop/4; i++ {
			for j := 0; j < pop/4; j++ {
				if i == j {
					continue
				}
				g := pool[i].Copy()
				w1 := pool[j].W1
				b1 := pool[j].B1
				w2 := pool[j].W2
				b2 := pool[j].B2
				g.W1[rng.Intn(len(g.W1))].Mean = w1[rng.Intn(len(w1))].Mean
				g.W1[rng.Intn(len(g.W1))].StdDev = w1[rng.Intn(len(w1))].StdDev
				g.B1[rng.Intn(len(g.B1))].Mean = b1[rng.Intn(len(b1))].Mean
				g.B1[rng.Intn(len(g.B1))].StdDev = b1[rng.Intn(len(b1))].StdDev
				g.W2[rng.Intn(len(g.W2))].Mean = w2[rng.Intn(len(w2))].Mean
				g.W2[rng.Intn(len(g.W2))].StdDev = w2[rng.Intn(len(w2))].StdDev
				g.B2[rng.Intn(len(g.B2))].Mean = b2[rng.Intn(len(b2))].Mean
				g.B2[rng.Intn(len(g.B2))].StdDev = b2[rng.Intn(len(b2))].StdDev
				pool = append(pool, g)
			}
		}
		for i := 0; i < pop; i++ {
			g := pool[i].Copy()
			g.W1[rng.Intn(len(g.W1))].Mean += rng.NormFloat64()
			g.W1[rng.Intn(len(g.W1))].StdDev += rng.NormFloat64()
			g.B1[rng.Intn(len(g.B1))].Mean += rng.NormFloat64()
			g.B1[rng.Intn(len(g.B1))].StdDev += rng.NormFloat64()
			g.W2[rng.Intn(len(g.W2))].Mean += rng.NormFloat64()
			g.W2[rng.Intn(len(g.W2))].StdDev += rng.NormFloat64()
			g.B2[rng.Intn(len(g.B2))].Mean += rng.NormFloat64()
			g.B2[rng.Intn(len(g.B2))].StdDev += rng.NormFloat64()
			pool = append(pool, g)
		}
		done := make(chan *AutoencoderNetwork, 8)
		i, flight := 0, 0
		task := func(rng *rand.Rand, i int) {
			_, stats, network, found := sample(rng, &pool[i])
			if found {
				done <- network
			}
			pool[i].Fitness = stats[0]
			pool[i].Cached = true
			done <- nil
		}
		for i < len(pool) && flight < cpus {
			if pool[i].Cached {
				i++
				continue
			}
			r := rngs[i]
			if r == nil {
				r = rand.New(rand.NewSource(rng.Int63()))
				rngs[i] = r
			}
			go task(r, i)
			i++
			flight++
		}
		for i < len(pool) {
			if pool[i].Cached {
				i++
				continue
			}

			if network := <-done; network != nil {
				network.Test(datum.Fisher)
				break Search
			}
			flight--

			r := rngs[i]
			if r == nil {
				r = rand.New(rand.NewSource(rng.Int63()))
				rngs[i] = r
			}
			go task(r, i)
			i++
			flight++
		}
		for flight > 0 {
			if network := <-done; network != nil {
				network.Test(datum.Fisher)
				break
			}
			flight--
		}
		generation++
	}
}

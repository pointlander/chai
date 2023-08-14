// Copyright 2023 The Chai Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"crypto/sha256"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"

	"github.com/pointlander/pagerank"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// ComplexProofOfWork implements a complex recurrent neural network for computing a proof of work
func ComplexProofOfWork(seed int) {
	cpus := runtime.NumCPU()
	rng := rand.New(rand.NewSource(int64(seed)))
	type Distribution struct {
		Mean   float64
		StdDev float64
	}
	const pop = 256
	const cols, rows = 256, 256
	const work = 26

	type Genome struct {
		A       []Distribution
		T       []Distribution
		Weights []Distribution
		Bias    []Distribution
		Fitness float64
		StdDev  float64
		Rank    float64
		Cached  bool
	}
	pool := make([]Genome, 0, pop)
	target := make([]byte, 0, 1024)
	for i := 0; i < 1024; i++ {
		target = append(target, byte(rng.Intn(256)))
	}
	target = target[:1]
	//target := []byte("And God said, Let there be light: and there was light.")
	size := len(target) * 8
	n := 32

	factor := math.Sqrt(2.0 / float64(cols))
	for i := 0; i < pop; i++ {
		weights := make([]Distribution, 0, cols*rows)
		for i := 0; i < cols*rows; i++ {
			weights = append(weights, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
		}
		bias := make([]Distribution, 0, rows)
		for i := 0; i < rows; i++ {
			bias = append(bias, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
		}
		a := make([]Distribution, 0, n)
		for i := 0; i < n; i++ {
			a = append(a, Distribution{Mean: rng.NormFloat64(), StdDev: rng.NormFloat64()})
		}
		t := make([]Distribution, 0, size)
		for i := 0; i < size; i++ {
			t = append(t, Distribution{Mean: rng.NormFloat64(), StdDev: rng.NormFloat64()})
		}
		g := Genome{
			A:       a,
			T:       t,
			Weights: weights,
			Bias:    bias,
		}
		pool = append(pool, g)
	}

	copy := func(g *Genome) Genome {
		a := make([]Distribution, len(g.A))
		copy(a, g.A)
		t := make([]Distribution, len(g.T))
		copy(t, g.T)
		weights := make([]Distribution, len(g.Weights))
		copy(weights, g.Weights)
		bias := make([]Distribution, len(g.Bias))
		copy(bias, g.Bias)
		return Genome{
			A:       a,
			T:       t,
			Weights: weights,
			Bias:    bias,
		}
	}

	sample := func(rng *rand.Rand, g *Genome) (samples plotter.Values, avg, stddev float64, found bool) {
		layer := NewMatrix(0, cols, rows)
		for _, w := range g.Weights {
			layer.Data = append(layer.Data, (rng.NormFloat64()+w.Mean)*w.StdDev)
		}
		b := NewMatrix(0, 1, rows)
		for _, w := range g.Bias {
			b.Data = append(b.Data, (rng.NormFloat64()+w.Mean)*w.StdDev)
		}
		inputs := NewMatrix(0, cols, 1)
		for i := 0; i < cols; i++ {
			if rng.Intn(2) == 0 {
				inputs.Data = append(inputs.Data, 1)
			} else {
				inputs.Data = append(inputs.Data, -1)
			}
		}
		cost, total := 0.0, 0.0
		var state uint32
		for i := 0; i < 128; i++ {
			for _, v := range g.A {
				targetCost := 0
				sampledT := make([]byte, 0, size)
				var buffer byte
				for i, v := range g.T {
					buffer <<= 1
					if (rng.NormFloat64()+v.Mean)*v.StdDev > 0 {
						buffer |= 1
					}
					if i%8 == 7 {
						sampledT = append(sampledT, buffer)
						for j := 0; j < 8; j++ {
							if (target[i/8]>>uint(j))&1 != (buffer>>uint(j))&1 {
								targetCost++
							}
						}
						buffer = 0
					}
				}

				if (rng.NormFloat64()+v.Mean)*v.StdDev > 0 {
					inputs.Data[0] = 1
				} else {
					inputs.Data[0] = -1
				}
				outputs := Add(Mul(layer, inputs), b)
				state <<= 1
				if outputs.Data[0] > 0 {
					state |= 1
				}
				input := make([]byte, 0, size)
				input = append(input, byte(state&0xff), byte((state>>8)&0xff), byte((state>>16)&0xff), byte((state>>24)&0xff))
				input = append(input, sampledT...)
				output := sha256.Sum256(input)
				iCost := 0
			Count:
				for _, v := range output {
					for j := 0; j < 8; j++ {
						if 0x80&(v<<uint(j)) == 0 {
							iCost++
						} else {
							break Count
						}
					}
				}
				iCost = 256 - iCost
				iCost += targetCost
				c := float64(iCost)
				cost += c
				stddev += c * c

				real := make([]byte, 0, size)
				real = append(real, byte(state&0xff), byte((state>>8)&0xff), byte((state>>16)&0xff), byte((state>>24)&0xff))
				real = append(real, target...)
				output = sha256.Sum256(real)
				iCost = 0
			Count2:
				for _, v := range output {
					for j := 0; j < 8; j++ {
						if 0x80&(v<<uint(j)) == 0 {
							iCost++
						} else {
							break Count2
						}
					}
				}
				if iCost >= work {
					found = true
					fmt.Println("found", iCost)
					break
				}

				for j := range outputs.Data {
					if outputs.Data[j] > 0 {
						outputs.Data[j] = 1
					} else {
						outputs.Data[j] = -1
					}
				}
				samples = append(samples, float64(iCost))
				inputs = outputs
			}
			total++
		}

		scale := total * float64(len(g.A))
		cost /= scale
		stddev /= scale
		squared := 0.0
		squared = cost * cost
		stddev -= squared
		stddev = math.Sqrt(stddev)
		return samples, cost, stddev, found
	}
	done := false
	d := make(plotter.Values, 0, 8)
	for i := range pool {
		dd, avg, stddev, found := sample(rng, &pool[i])
		fmt.Println(i, avg, stddev)
		if found {
			done = true
			break
		}
		pool[i].Fitness = avg
		pool[i].StdDev = stddev
		pool[i].Cached = true
		d = append(d, dd...)
	}

	p := plot.New()
	p.Title.Text = "rnn"

	histogram, err := plotter.NewHist(d, 10)
	if err != nil {
		panic(err)
	}
	p.Add(histogram)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "proof_of_work.png")
	if err != nil {
		panic(err)
	}

	rngs, generation := make(map[int]*rand.Rand), 0
Search:
	for !done {
		graph := pagerank.NewGraph64()
		for i := range pool {
			for j := i + 1; j < len(pool); j++ {
				// http://homework.uoregon.edu/pub/class/es202/ztest.html
				avga := pool[i].Fitness
				avgb := pool[j].Fitness
				avg := avga - avgb
				if avg < 0 {
					avg = -avg
				}
				stddeva := pool[i].StdDev
				stddevb := pool[j].StdDev
				stddev := math.Sqrt(stddeva*stddeva + stddevb*stddevb)
				z := stddev / avg
				graph.Link(uint64(i), uint64(j), z)
				graph.Link(uint64(j), uint64(i), z)
			}
		}
		graph.Rank(0.85, 0.000001, func(node uint64, rank float64) {
			pool[node].Rank = rank
		})
		sort.Slice(pool, func(i, j int) bool {
			//return pool[i].Fitness < pool[j].Fitness
			return pool[i].Rank > pool[j].Rank
		})
		pool = pool[:pop]
		fmt.Println(generation, pool[0].Fitness, pool[0].StdDev)
		if pool[0].Fitness < 1e-32 {
			break Search
		}
		for i := 0; i < pop/4; i++ {
			for j := 0; j < pop/4; j++ {
				if i == j {
					continue
				}
				g := copy(&pool[i])
				aa := pool[j].A
				tt := pool[j].T
				w := pool[j].Weights
				b := pool[j].Bias
				g.A[rng.Intn(len(g.A))].Mean = aa[rng.Intn(len(aa))].Mean
				g.A[rng.Intn(len(g.A))].StdDev = aa[rng.Intn(len(aa))].StdDev
				g.T[rng.Intn(len(g.T))].Mean = tt[rng.Intn(len(tt))].Mean
				g.T[rng.Intn(len(g.T))].StdDev = tt[rng.Intn(len(tt))].StdDev
				g.Weights[rng.Intn(len(g.Weights))].Mean = w[rng.Intn(len(w))].Mean
				g.Weights[rng.Intn(len(g.Weights))].StdDev = w[rng.Intn(len(w))].StdDev
				g.Bias[rng.Intn(len(g.Bias))].Mean = b[rng.Intn(len(b))].Mean
				g.Bias[rng.Intn(len(g.Bias))].StdDev = b[rng.Intn(len(b))].StdDev
				pool = append(pool, g)
			}
		}
		for i := 0; i < pop; i++ {
			g := copy(&pool[i])
			g.A[rng.Intn(len(g.A))].Mean += rng.NormFloat64()
			g.A[rng.Intn(len(g.A))].StdDev += rng.NormFloat64()
			g.T[rng.Intn(len(g.T))].Mean += rng.NormFloat64()
			g.T[rng.Intn(len(g.T))].StdDev += rng.NormFloat64()
			g.Weights[rng.Intn(len(g.Weights))].Mean += rng.NormFloat64()
			g.Weights[rng.Intn(len(g.Weights))].StdDev += rng.NormFloat64()
			g.Bias[rng.Intn(len(g.Bias))].Mean += rng.NormFloat64()
			g.Bias[rng.Intn(len(g.Bias))].StdDev += rng.NormFloat64()
			pool = append(pool, g)
		}
		done := make(chan bool, 8)
		i, flight := 0, 0
		task := func(rng *rand.Rand, i int) {
			_, avg, stddev, found := sample(rng, &pool[i])
			if found {
				done <- true
			}
			pool[i].Fitness = avg
			pool[i].StdDev = stddev
			pool[i].Cached = true
			done <- false
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

			if <-done {
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
			<-done
			flight--
		}
		generation++
	}
}

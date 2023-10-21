// Copyright 2023 The Chai Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/big"
	"math/rand"
	"runtime"
	"sort"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

//time ./chai -rnn -number 340282366920938460843936948965011886881
//...
//18446744073709551533 18446744073709551557
//
//real    97m36.142s
//user    187m3.345s
//sys     0m41.107s

//time ./chai -rnn -number 340282366920938460843936948965011886881
//...
//18446744073709551533 18446744073709551557
//
//real    48m16.379s
//user    94m20.563s
//sys     0m20.269s

// RNN implements a recurrent neural network for factoring integers
func RNN(seed int) {
	cpus := runtime.NumCPU()
	rnd := rand.New(rand.NewSource(int64(seed)))
	type Distribution struct {
		Mean   float64
		StdDev float64
	}
	const pop = 256
	const cols, rows = 32, 32
	zero := big.Int{}
	zero.SetUint64(0)
	one := big.Int{}
	one.SetUint64(1)
	two := big.Float{}
	two.SetUint64(2)

	shownum := func(a []Distribution) big.Int {
		x := big.Int{}
		e := big.Int{}
		e.SetInt64(1)
		for _, v := range a {
			if v.Mean > 0 {
				x.Add(&x, &e)
			}
			e.Lsh(&e, 1)
		}
		return x
	}

	type Genome struct {
		A       []Distribution
		T       []Distribution
		Weights []Distribution
		Bias    []Distribution
		Fitness big.Float
		StdDev  big.Float
		Rank    float64
		Cached  bool
	}
	pool := make([]Genome, 0, pop)
	target := big.Int{}
	target.SetString(*FlagNumber, 10)
	size := target.BitLen()
	nn := big.Float{}
	nn.SetInt(&target)
	nn.Sqrt(&nn)
	n := nn.MantExp(nil)

	factor := math.Sqrt(2.0 / float64(cols))
	for i := 0; i < pop; i++ {
		weights := make([]Distribution, 0, cols*rows)
		for i := 0; i < cols*rows; i++ {
			weights = append(weights, Distribution{Mean: factor * rnd.NormFloat64(), StdDev: factor * rnd.NormFloat64()})
		}
		bias := make([]Distribution, 0, rows)
		for i := 0; i < rows; i++ {
			bias = append(bias, Distribution{Mean: factor * rnd.NormFloat64(), StdDev: factor * rnd.NormFloat64()})
		}
		a := make([]Distribution, 0, n)
		for i := 0; i < n; i++ {
			a = append(a, Distribution{Mean: rnd.NormFloat64(), StdDev: rnd.NormFloat64()})
		}
		t := make([]Distribution, 0, size)
		for i := 0; i < size; i++ {
			t = append(t, Distribution{Mean: rnd.NormFloat64(), StdDev: rnd.NormFloat64()})
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

	sample := func(rng *rand.Rand, g *Genome) (samples plotter.Values, avg, stddev big.Float, found bool, percent float64) {
		mask := big.Int{}
		mask.SetUint64(1)
		mask.Lsh(&mask, uint(n+1))
		mask.Sub(&mask, &one)

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
		state := big.Int{}
		cost, total, count, index := big.Float{}, 0.0, 0.0, 0
		costs := make([]big.Float, 32*len(g.A))
		maxDensity := func() (int, big.Float) {
			sort.Slice(costs, func(i, j int) bool {
				return costs[i].Cmp(&costs[j]) < 0
			})
			index, min, cost := 0, big.Float{}, big.Float{}
			for i := range costs[:len(costs)-8] {
				mean, size := big.Float{}, big.Float{}
				for j := i; j < i+8; j++ {
					mean.Add(&mean, &costs[j])
					size.Add(&size, big.NewFloat(1))
				}
				mean.Quo(&mean, &size)
				stddev := big.Float{}
				for j := i; j < i+8; j++ {
					diff := big.Float{}
					diff.Sub(&mean, &costs[j])
					diff.Mul(&diff, &diff)
					stddev.Add(&stddev, &diff)
				}
				stddev.Quo(&stddev, &size)
				stddev.Sqrt(&stddev)
				if i == 0 || stddev.Cmp(&min) < 0 {
					min.Copy(&stddev)
					index = i + 4
					cost.Copy(&costs[index])
				}
			}
			return index, cost
		}
		for i := 0; i < 32; i++ {
			sampledT := big.Int{}
			e := big.Int{}
			e.SetInt64(1)
			for _, v := range g.T {
				if (rng.NormFloat64()+v.Mean)*v.StdDev > 0 {
					sampledT.Add(&sampledT, &e)
				}
				e.Lsh(&e, 1)
			}

			targetCost := big.Int{}
			targetCost.Sub(&target, &sampledT)
			targetCost.Abs(&targetCost)
			for _, v := range g.A {
				if (rng.NormFloat64()+v.Mean)*v.StdDev > 0 {
					inputs.Data[0] = 1
				} else {
					inputs.Data[0] = -1
				}
				outputs := Add(Mul(layer, inputs), b)
				state.Lsh(&state, 1)
				if outputs.Data[0] > 0 {
					state.Or(&state, &one)
				}
				state.And(&state, &mask)
				if state.ProbablyPrime(100) {
					percent++
				}
				count++
				if state.Cmp(&zero) != 0 && state.Cmp(&one) != 0 {
					mod := big.Int{}
					mod.Mod(&target, &state)
					if mod.Cmp(&zero) == 0 {
						div := big.Int{}
						div.Div(&target, &state)
						fmt.Println(state.String(), div.String())
						found = true
						scale := big.NewFloat((total * float64(len(g.A))))
						cost.Quo(&cost, scale)
						stddev.Quo(&stddev, scale)
						squared := big.Float{}
						squared.Mul(&cost, &cost)
						stddev.Sub(&stddev, &squared)
						stddev.Sqrt(&stddev)
						costs = costs[:index]
						_, min := maxDensity()
						return samples, min, stddev, found, percent / count
					}
				}
				iCost := big.Int{}
				if state.Cmp(&zero) != 0 {
					iCost.Mod(&sampledT, &state)
				} else {
					iCost.Set(&sampledT)
				}
				iCost.Abs(&iCost)
				iCost.Add(&iCost, &targetCost)
				denom := big.Float{}
				denom.SetInt(&target)
				denom.Mul(&denom, &two)
				num := big.Float{}
				num.SetInt(&iCost)
				num.Quo(&num, &denom)
				costs[index].Copy(&cost)
				index++
				cost.Add(&cost, &num)
				num.Mul(&num, &num)
				stddev.Add(&stddev, &num)
				for j := range outputs.Data {
					if outputs.Data[j] > 0 {
						outputs.Data[j] = 1
					} else {
						outputs.Data[j] = -1
					}
				}
				s, _ := num.Float64()
				samples = append(samples, s)
				inputs = outputs
			}
			total++
		}

		scale := big.NewFloat((total * float64(len(g.A))))
		cost.Quo(&cost, scale)
		stddev.Quo(&stddev, scale)
		squared := big.Float{}
		squared.Mul(&cost, &cost)
		stddev.Sub(&stddev, &squared)
		stddev.Sqrt(&stddev)
		_, min := maxDensity()
		return samples, min, stddev, found, percent / count
	}
	done := false
	d := make(plotter.Values, 0, 8)
	for i := range pool {
		dd, avg, stddev, found, count := sample(rnd, &pool[i])
		fmt.Println(i, count, avg.String(), stddev.String())
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

	err = p.Save(8*vg.Inch, 8*vg.Inch, "rnn.png")
	if err != nil {
		panic(err)
	}

	rngs := make([]*rand.Rand, cpus)
	for i := range rngs {
		rngs[i] = rand.New(rand.NewSource(int64(i + 1)))
	}

Search:
	for !done {
		/*graph := pagerank.NewGraph64()
		for i := range pool {
			for j := i + 1; j < len(pool); j++ {
				// http://homework.uoregon.edu/pub/class/es202/ztest.html
				avga, _ := pool[i].Fitness.Float64()
				avgb, _ := pool[j].Fitness.Float64()
				avg := avga - avgb
				if avg < 0 {
					avg = -avg
				}
				stddeva, _ := pool[i].StdDev.Float64()
				stddevb, _ := pool[j].StdDev.Float64()
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
			return pool[i].Fitness.Cmp(&pool[j].Fitness) < 0
			//return pool[i].Rank > pool[j].Rank
		})
		pool = pool[:pop]
		fmt.Println(pool[0].Fitness.String(), pool[0].StdDev.String())
		for i := range pool {
			x := shownum(pool[i].A)
			if x.Cmp(&zero) == 0 {
				continue
			}
			mod := big.Int{}
			mod.Mod(&target, &x)
			if mod.Cmp(&zero) == 0 {
				if x.Cmp(&one) == 1 || x.Cmp(&target) == 0 {
					continue
				} else {
					div := big.Int{}
					div.Div(&target, &x)
					fmt.Println(x.String(), div.String())
					break Search
				}
			}
		}
		if pool[0].Fitness.Cmp(big.NewFloat(1e-32)) < 0 {
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
				g.A[rnd.Intn(len(g.A))].Mean = aa[rnd.Intn(len(aa))].Mean
				g.A[rnd.Intn(len(g.A))].StdDev = aa[rnd.Intn(len(aa))].StdDev
				g.T[rnd.Intn(len(g.T))].Mean = tt[rnd.Intn(len(tt))].Mean
				g.T[rnd.Intn(len(g.T))].StdDev = tt[rnd.Intn(len(tt))].StdDev
				g.Weights[rnd.Intn(len(g.Weights))].Mean = w[rnd.Intn(len(w))].Mean
				g.Weights[rnd.Intn(len(g.Weights))].StdDev = w[rnd.Intn(len(w))].StdDev
				g.Bias[rnd.Intn(len(g.Bias))].Mean = b[rnd.Intn(len(b))].Mean
				g.Bias[rnd.Intn(len(g.Bias))].StdDev = b[rnd.Intn(len(b))].StdDev
				pool = append(pool, g)
			}
		}
		for i := 0; i < pop; i++ {
			g := copy(&pool[i])
			g.A[rnd.Intn(len(g.A))].Mean += rnd.NormFloat64()
			g.A[rnd.Intn(len(g.A))].StdDev += rnd.NormFloat64()
			g.T[rnd.Intn(len(g.T))].Mean += rnd.NormFloat64()
			g.T[rnd.Intn(len(g.T))].StdDev += rnd.NormFloat64()
			g.Weights[rnd.Intn(len(g.Weights))].Mean += rnd.NormFloat64()
			g.Weights[rnd.Intn(len(g.Weights))].StdDev += rnd.NormFloat64()
			g.Bias[rnd.Intn(len(g.Bias))].Mean += rnd.NormFloat64()
			g.Bias[rnd.Intn(len(g.Bias))].StdDev += rnd.NormFloat64()
			pool = append(pool, g)
		}
		done := make(chan *rand.Rand, 8)
		i, flight := 0, 0
		task := func(rng *rand.Rand, i int) {
			_, avg, stddev, found, _ := sample(rng, &pool[i])
			if found {
				done <- nil
			}
			pool[i].Fitness = avg
			pool[i].StdDev = stddev
			pool[i].Cached = true
			done <- rng
		}
		for i < len(pool) && flight < cpus {
			if pool[i].Cached {
				i++
				continue
			}
			go task(rngs[flight], i)
			i++
			flight++
		}
		for i < len(pool) {
			if pool[i].Cached {
				i++
				continue
			}

			rng := <-done
			if rng == nil {
				break Search
			}
			flight--

			go task(rng, i)
			i++
			flight++
		}
		for flight > 0 {
			<-done
			flight--
		}
	}
}

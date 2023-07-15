// Copyright 2023 The Chai Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"sort"
)

var (
	// FlagBits bits mode
	FlagBits = flag.Bool("bits", false, "bits mode")
	// FlagNumbers numbers mode
	FlagNumbers = flag.Bool("numbers", false, "numbers mode")
	// FlagSwarm swarm mode
	FlagSwarm = flag.Bool("swarm", false, "swarm mode")
)

func main() {
	flag.Parse()

	if *FlagBits {
		Bits()
		return
	}

	if *FlagNumbers {
		Numbers()
		return
	}

	if *FlagSwarm {
		Swarm()
		return
	}
}

func Bits() {
	rnd := rand.New(rand.NewSource(1))
	type Distribution struct {
		Mean   float64
		StdDev float64
	}
	target := 77
	n := int(math.Ceil(math.Log2(float64(target))))
	a := make([]Distribution, 0, n)
	b := make([]Distribution, 0, n)
	for i := 0; i < n; i++ {
		a = append(a, Distribution{Mean: 0, StdDev: 1})
		b = append(b, Distribution{Mean: 0, StdDev: 1})
	}
	fmt.Println(n)
	fmt.Println(a)
	fmt.Println(b)
	shownum := func(a []Distribution) {
		x := 0.0
		e := 1.0
		for _, v := range a {
			if v.Mean > 0 {
				x += e
			}
			e *= 2
		}
		fmt.Println(x)
	}
	samples := 8 * 1024
	sample := func(a, b []Distribution, ia, ib int, d float64) (avg, sd float64) {
		i := 0
		for i < samples {
			x := 0
			y := 0
			e := 1
			k := 0
			for _, v := range a {
				if k == ia {
					v.Mean = v.Mean + d
				}
				if (rnd.NormFloat64()+v.Mean)*v.StdDev > 0 {
					x += e
				}
				e *= 2
				k++
			}
			e = 1
			k = 0
			for _, v := range b {
				if k == ib {
					v.Mean = v.Mean + d
				}
				if (rnd.NormFloat64()+v.Mean)*v.StdDev > 0 {
					y += e
				}
				e *= 2
				k++
			}
			xx := 0
			if x > 0 {
				xx = target % x
			}
			yy := 0
			if y > 0 {
				yy = target % y
			}
			cost := target - x*y
			if cost < 0 {
				cost = -cost
			}
			cost += yy + xx
			avg += float64(cost)
			sd += float64(cost) * float64(cost)
			i += 1
		}
		avg /= float64(samples)
		sd = math.Sqrt(sd/float64(samples) - avg*avg)
		return avg, sd
	}

	avg, sd := sample(a, b, -1, -1, 0)
	j := 0
	for j < 1000 {
		if sd == 0 {
			break
		}
		type Sample struct {
			DeltasA []Distribution
			DeltasB []Distribution
			Avg     float64
			Sd      float64
		}
		samples := make([]Sample, 16)
		for i := range samples {
			samples[i].DeltasA = make([]Distribution, len(a))
			copy(samples[i].DeltasA, a)
			for j := range samples[i].DeltasA {
				samples[i].DeltasA[j].Mean += rnd.NormFloat64()
				samples[i].DeltasA[j].StdDev += rnd.NormFloat64()
				if samples[i].DeltasA[j].StdDev < 0 {
					samples[i].DeltasA[j].StdDev = -samples[i].DeltasA[j].StdDev
				}
			}
			samples[i].DeltasB = make([]Distribution, len(b))
			copy(samples[i].DeltasB, b)
			for j := range samples[i].DeltasB {
				samples[i].DeltasB[j].Mean += rnd.NormFloat64()
				samples[i].DeltasB[j].StdDev += rnd.NormFloat64()
				if samples[i].DeltasB[j].StdDev < 0 {
					samples[i].DeltasB[j].StdDev = -samples[i].DeltasB[j].StdDev
				}
			}
			samples[i].Avg, samples[i].Sd = sample(samples[i].DeltasA, samples[i].DeltasB, -1, -1, 0)
		}
		sort.Slice(samples, func(i, j int) bool {
			return samples[i].Avg < samples[j].Avg
		})
		avg = samples[1].Avg
		sd = samples[1].Sd
		a = samples[1].DeltasA
		b = samples[1].DeltasB
		fmt.Println(j, avg, sd)
		j++
	}
	fmt.Println(a)
	fmt.Println(b)
	shownum(a)
	shownum(b)
}

func Swarm() {
	rnd := rand.New(rand.NewSource(1))
	type Distribution struct {
		Mean   float64
		StdDev float64
	}
	target := 77
	max := math.Sqrt(float64(target))
	n := int(math.Ceil(math.Log2(max)))

	shownum := func(a []Distribution) int {
		x := 0
		e := 1
		for _, v := range a {
			if v.Mean > 0 {
				x += e
			}
			e *= 2
		}
		fmt.Println(x)
		return x
	}

	samples := 16 * 1024
	sample := func(a, b []Distribution) (avg, sd float64) {
		i := 0
		for i < samples {
			x := 0
			y := 0
			e := 1
			for _, v := range a {
				if (rnd.NormFloat64()+v.Mean)*v.StdDev > 0 {
					x += e
				}
				e *= 2
			}
			e = 1
			for _, v := range b {
				if (rnd.NormFloat64()+v.Mean)*v.StdDev > 0 {
					y += e
				}
				e *= 2
			}
			xx := 0
			if x > 0 {
				xx = target % x
			}
			yy := 0
			if y > 0 {
				yy = target % y
			}
			cost := target - x*y
			if cost < 0 {
				cost = -cost
			}
			cost += yy + xx
			avg += float64(cost)
			sd += float64(cost) * float64(cost)
			i += 1
		}
		avg /= float64(samples)
		sd = math.Sqrt(sd/float64(samples) - avg*avg)
		return avg, sd
	}

	g := math.MaxFloat64
	g1 := make([]Distribution, n)
	g2 := make([]Distribution, n)
	type Particle struct {
		X []Distribution
		P []Distribution
		F float64
		V []float64
	}
	particles := make([]Particle, 16)
	for i := range particles {
		a := make([]Distribution, 0, n)
		for i := 0; i < n; i++ {
			a = append(a, Distribution{Mean: (2*rnd.Float64() - 1) * 10, StdDev: 1})
		}
		particles[i].X = a
		x := make([]Distribution, n)
		copy(x, a)
		particles[i].P = x
		v := make([]float64, n)
		for i := range v {
			v[i] = (2*rnd.Float64() - 1) * 10
		}
		particles[i].V = v
	}

	for i := range particles {
		self := particles[i:]
		for j := range self {
			a, _ := sample(particles[i].P, self[j].P)
			if a < g {
				particles[i].F = a
				self[j].F = a
				g = a
				copy(g1, particles[i].P)
				copy(g2, self[j].P)
			}
		}
	}

	const w, w1, w2 = .5, .8, .1
	for {
		for i := range particles {
			for j := range particles[i].X {
				rp, rg := rnd.Float64(), rnd.Float64()
				particles[i].V[j] = w*particles[i].V[j] +
					w1*rp*(particles[i].P[j].Mean-particles[i].X[j].Mean) +
					w2*rg*(g1[j].Mean-particles[i].X[j].Mean)
			}
			for j := range particles[i].X {
				particles[i].X[j].Mean += particles[i].V[j]
			}
		}
		for i := range particles {
			self := particles[i:]
			for j := range self {
				a, s := sample(particles[i].X, self[j].X)
				if a < particles[i].F || a < particles[j].F {
					copy(particles[i].P, particles[i].X)
					copy(self[j].P, self[j].X)
					particles[i].F = a
					self[j].F = a
					if a < g {
						g = a
						fmt.Println(g, s)
						copy(g1, particles[i].P)
						copy(g2, self[j].P)
						x := shownum(g1)
						y := shownum(g2)
						if x*y == target {
							return
						}
					}
				}
			}
		}
	}
}

func Numbers() {
	rnd := rand.New(rand.NewSource(1))
	type Distribution struct {
		Mean   float64
		StdDev float64
	}
	target := 77
	n := int(math.Ceil(math.Log2(float64(target))))
	a := Distribution{Mean: 0, StdDev: math.Ceil(math.Sqrt(float64(target)))}
	b := Distribution{Mean: 0, StdDev: math.Ceil(math.Sqrt(float64(target)))}
	fmt.Println(n)
	fmt.Println(a)
	fmt.Println(b)
	shownum := func(a Distribution) {
		fmt.Println(a.Mean)
	}
	samples := 8 * 1024
	sample := func(a, b Distribution, ia, ib int, d float64) (avg, sd float64) {
		i := 0
		for i < samples {
			x := int((rnd.NormFloat64() + a.Mean) * a.StdDev)
			y := int((rnd.NormFloat64() + b.Mean) * b.StdDev)
			xx := 0
			if x > 0 {
				xx = target % x
			}
			yy := 0
			if y > 0 {
				yy = target % y
			}
			cost := target - x*y
			if cost < 0 {
				cost = -cost
			}
			cost += yy + xx
			avg += float64(cost)
			sd += float64(cost) * float64(cost)
			i += 1
		}
		avg /= float64(samples)
		sd = math.Sqrt(sd/float64(samples) - avg*avg)
		return avg, sd
	}

	avg, sd := sample(a, b, -1, -1, 0)
	j := 0
	for j < 1000 {
		if sd == 0 {
			break
		}
		type Sample struct {
			DeltasA Distribution
			DeltasB Distribution
			Avg     float64
			Sd      float64
		}
		samples := make([]Sample, 16)
		for i := range samples {
			samples[i].DeltasA = Distribution{Mean: a.Mean + rnd.NormFloat64(), StdDev: a.StdDev + rnd.NormFloat64()}
			if samples[i].DeltasA.StdDev < 0 {
				samples[i].DeltasA.StdDev = -samples[i].DeltasA.StdDev
			}
			samples[i].DeltasB = Distribution{Mean: b.Mean + rnd.NormFloat64(), StdDev: b.StdDev + rnd.NormFloat64()}
			if samples[i].DeltasB.StdDev < 0 {
				samples[i].DeltasB.StdDev = -samples[i].DeltasB.StdDev
			}
			samples[i].Avg, samples[i].Sd = sample(samples[i].DeltasA, samples[i].DeltasB, -1, -1, 0)
		}
		sort.Slice(samples, func(i, j int) bool {
			return samples[i].Avg < samples[j].Avg
		})
		avg = samples[0].Avg
		sd = samples[0].Sd
		a = samples[0].DeltasA
		b = samples[0].DeltasB
		fmt.Println(j, avg, sd)
		j++
	}
	fmt.Println(a)
	fmt.Println(b)
	shownum(a)
	shownum(b)
}

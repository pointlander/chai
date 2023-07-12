// Copyright 2023 The Chai Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

func main() {
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

// Copyright 2023 The Chai Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
)

func main() {
	rnd := rand.New(rand.NewSource(1))
	target := 77
	n := int(math.Ceil(math.Log2(float64(target))))
	a := make([]float64, 0, n)
	b := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		a = append(a, 0)
		b = append(b, 0)
	}
	fmt.Println(n)
	fmt.Println(a)
	fmt.Println(b)
	shownum := func(a []float64) {
		x := 0.0
		e := 1.0
		for _, v := range a {
			if v > 0 {
				x += e
			}
			e *= 2
		}
		fmt.Println(x)
	}
	costs := []int{}
	samples := 8 * 1024
	sample := func(a, b []float64, ia, ib int, d float64) (avg, sd float64) {
		i := 0
		for i < samples {
			x := 0
			y := 0
			e := 1
			k := 0
			for _, v := range a {
				if k == ia {
					v = v + d
				}
				if (rnd.NormFloat64() + v) > 0 {
					x += e
				}
				e *= 2
				k++
			}
			e = 1
			k = 0
			for _, v := range b {
				if k == ib {
					v = v + d
				}
				if (rnd.NormFloat64() + v) > 0 {
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
			if d == 0 {
				costs = append(costs, cost)
			}
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
		min1 := 0.0
		min2 := 0.0
		index1 := -1
		index2 := -1
		for i := 0; i < n; i++ {
			step1 := rnd.NormFloat64()
			aa := make([]float64, len(a))
			copy(aa, a)
			aa[i] += step1
			for j := 0; j < n; j++ {
				step2 := rnd.NormFloat64()
				x, sx := sample(aa, b, -1, j, step2)
				y, sy := sample(aa, b, -1, j, -step2)
				if x < avg {
					avg = x
					sd = sx
					min1 = step1
					min2 = step2
					index1 = i
					index2 = j
				}
				if y < avg {
					avg = y
					sd = sy
					min1 = step1
					min2 = -step2
					index1 = i
					index2 = j
				}
			}

			copy(aa, a)
			aa[i] -= step1
			for j := 0; j < n; j++ {
				step2 := rnd.NormFloat64()
				x, sx := sample(aa, b, -1, j, step2)
				y, sy := sample(aa, b, -1, j, -step2)
				if x < avg {
					avg = x
					sd = sx
					min1 = -step1
					min2 = step2
					index1 = i
					index2 = j
				}
				if y < avg {
					avg = y
					sd = sy
					min1 = -step1
					min2 = -step2
					index1 = i
					index2 = j
				}
			}
		}
		if index1 >= 0 && index2 >= 0 {
			a[index1] += min1
			b[index2] += min2
			fmt.Println(j, avg, sd)
		}
		j++
	}
	fmt.Println(a)
	fmt.Println(b)
	shownum(a)
	shownum(b)
}

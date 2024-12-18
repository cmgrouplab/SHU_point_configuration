////////////////////////////////////////////////////////////////////////////////
// stealthy.cpp --- "Generates hyperuniform stealhy patterns"
////////////////////////////////////////////////////////////////////////////////
// This code is part of the Supplemental Material for:
// L.S. Froufe-Perez, M. Engel, P.F. Damasceno, N. Muller, J. Haberko, S.C. Glotzer, and F. Scheffold, "Role of Short-Range Order and Hyperuniformity in the Formation of Band Gaps in Disordered Photonic Materials" (2016).
// written by Michael Engel (University of Michigan) on 15 July 2015
//
// Compilation:
//    g++ stealthy.cpp -O3 -o stealthy
//
// Example usages:
//    echo "0.3 200 1.0 1e-1 1e-7 10000 0" | ./stealthy > pattern.chi0.3.pos
//    for i in `seq 1 8`; do echo "0.$i 200 1.0 1e-1 1e-7 10000 0" | ./stealthy > pattern.chi0.$i.pos; done
//
// Parameters in the example:
//    0.5   : chi parameter for the stealthy pattern
//    200   : target value for number of particles (can slightly change)
//    1e-1  : start value for kinetic energy
//    1e-7  : end value for kinetic energy
//    1e4   : number of molecular dynamics steps during simulated annealing
//    0     : seed for random number generator
//
// Notes:
// (1) The code uses a simulated annealing scheme to find an equilibrium
//     stealthy hyperuniform pattern. This is not the fastest way but turns out
//     to be quite robust.
// (2) The density is chosen such that the k-cutoff is at k = 2 Pi
// (3) The algorithm slows down with O(N^2). I typically run <= 1000 particles.
// (4) The structure formation occurs in the energy range [1e-1, 1e-3]. Below
//     that range the system mostly relaxes locally. The code can be sped up by
//     stoping the relaxation at a higher energy than 1e-7.
// (5) At high chi >= 0.6 the system tends to gets stuck in crystalline states.
//     If better relaxation is needed then the energy scale has to be adjusted.
// (6) For production runs values of MDSASTEPS >= 10000 are recommended. At
//     higher chi values larger MDSASTEPS improves relaxation.
////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <iostream>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Global variables
////////////////////////////////////////////////////////////////////////////////

// initial values:
double CHI          = 0.5;
int    TARGETNUMPAR = 200;
double EKINSTART    = 1e-1;
double EKINEND      = 1e-7;
int    MDSASTEPS    = 10000;
int    RNGSEED      = 0;

// not changeable from command line
const double TIMESTEP     = 0.1;
const int    OUTPUTNUM    = 100;
const int    VERLETUPDATE = 100;

////////////////////////////////////////////////////////////////////////////////
// 2D vector class
////////////////////////////////////////////////////////////////////////////////

class Vector2d
{
public:
	
	double x;
	double y;
	
	Vector2d()
	{
		x = 0.0;
		y = 0.0;
	}
	
	Vector2d(const double& r, const double& s)
	{
		x = r;
		y = s;
	}
	
	Vector2d& operator =(const Vector2d& v)
	{
		x = v.x;
		y = v.y;
		return *this;
	}
	
	Vector2d& operator +=(const Vector2d& v)
	{
		x += v.x;
		y += v.y;
		return (*this);
	}
	
	Vector2d& operator -=(const Vector2d& v)
	{
		x -= v.x;
		y -= v.y;
		return (*this);
	}
	
	Vector2d& operator *=(const double& t)
	{
		x *= t;
		y *= t;
		return (*this);
	}
	
	Vector2d& operator /=(const double& t)
	{
		return (*this *= (1.0 / t));
	}
	
	const Vector2d operator -(void) const
	{
		return (Vector2d(-x, -y));
	}
	
	const Vector2d operator +(const Vector2d& v) const
	{
		return (Vector2d(*this) += v);
	}
	
	const Vector2d operator -(const Vector2d& v) const
	{
		return (Vector2d(*this) -= v);
	}
	
	const Vector2d operator *(const double& t) const
	{
		return (Vector2d(*this) *= t);
	}
	
	const Vector2d operator /(const double& t) const
	{
		return (Vector2d(*this) /= t);
	}
	
	double operator *(const Vector2d& v) const
	{
		return (x * v.x + y * v.y);
	}
};

const Vector2d operator *(const double& t, const Vector2d& v)
{
	return (v * t);
}

////////////////////////////////////////////////////////////////////////////////
// Data structures and helper functions
////////////////////////////////////////////////////////////////////////////////

// particle data
struct Particle
{
	// position, velocity, acceleration, old acceleration
	Vector2d x, v, a, aOld;
	// cos sum and sin sum temporary storage
	double c, s;
};
std::vector<Particle> particles;

// simulation box
Vector2d box;

// k-vectors that are excluded (= used for the photonic gap)
std::vector<Vector2d> kTable;

// Gaussian distributed random number (variance s) with Box-Muller transform
double randGauss(double s)
{
	double x1, x2, w;
	do
	{
		x1 = 2.0 * drand48() - 1.0;
		x2 = 2.0 * drand48() - 1.0;
		w = x1 * x1 + x2 * x2;
	}
	while (w >= 1.0);
	return x1 * sqrt(-2.0 * log(w) / w) * s;
}

////////////////////////////////////////////////////////////////////////////////
// Main function
////////////////////////////////////////////////////////////////////////////////

int main()
{
	// input parameters
	std::cin >> CHI;
	std::cin >> TARGETNUMPAR;
	std::cin >> EKINSTART;
	std::cin >> EKINEND;
	std::cin >> MDSASTEPS;
	std::cin >> RNGSEED;

	// number of constraint k-vectors targeted
	double m_k = CHI * 2 * (TARGETNUMPAR - 1);
	// excluded radius in k-space
	//    M_PI * kRange^2 < 2 * m_k * (kx * ky)
	double kRangeSqr = m_k * 2.0 / M_PI;

	// generate table of k-vectors within a disk
	// loop over all reciprocal lattice points in excluded disk
	int kRi = (int)ceil(sqrt(kRangeSqr));
	for (int ikx =    0; ikx <= kRi; ikx++)
	for (int iky = -kRi; iky <= kRi; iky++)
	{
		// avoid double-counting
		if (ikx == 0.0 && iky <= 0.0) continue;
		// if within cut-off range...
		if (ikx * ikx + iky * iky < kRangeSqr)
		{
			kTable.push_back(Vector2d(ikx, iky));
		}
	}

	// adjust number of particles
	int numParticles = (double)kTable.size() / (CHI * 2) + 1;

	// generate box
	// keep k cutoff at (2 Pi), i.e.:  (m_k * 2 + 1) / boxSize^2 = Pi
	double boxSize = sqrt((CHI * 4.0 * (numParticles - 1) + 1) / M_PI);
	box = Vector2d(boxSize, boxSize);
	
	// rescale k vectors
	for (int k = 0; k < kTable.size(); k++)
	{
		Vector2d& kv = kTable[k];
		kv.x *= 2.0 * M_PI / box.x;
		kv.y *= 2.0 * M_PI / box.y;
	}
	
	// initialize the random number generator
	srand48(RNGSEED);
	// place particles at random positions in the box
	for (int i = 0; i < numParticles; i++)
	{
		Particle p;
		p.x.x = drand48() * box.x;
		p.x.y = drand48() * box.y;
		particles.push_back(p);
	}
	
	////////////////////////////////////////////////////////////////////////////////
	// MD Simulation: relaxation via simulated annealing
	////////////////////////////////////////////////////////////////////////////////
	
	for (int t = 0; t <= MDSASTEPS; t++)
	{
		// Use 'velocity verlet' algorithm; half step velocity step is eliminated
		// STEP 1: update position
		//    x(t + dt) = x(t) + v(t) * dt + 0.5 * a(t) * dt^2
		for (int i = 0; i < particles.size(); i++)
		{
			Particle& p = particles[i];
			p.x += TIMESTEP * p.v + (0.5 * TIMESTEP * TIMESTEP) * p.a;
			p.a = Vector2d();
		}

		// STEP 2: calculate acceleration (assume: mass = 1)
		// set energy scale to get rid of system size effects
		double scale = 1.0 / kTable.size();
		double epot = 0.0;
		for (int k = 0; k < kTable.size(); k++)
		{
			const Vector2d& kv = kTable[k];
			
			// precalculate sum
			double fc = 0.0, fs = 0.0;
			for (int i = 0; i < particles.size(); i++)
			{
				Particle& p = particles[i];
				double arg = p.x * kv;
				p.c = cos(arg);
				p.s = sin(arg);
				fc += p.c;
				fs += p.s;
			}
			fc *= scale;
			fs *= scale;
			epot += fc * fc + fs * fs;
			
			// add contributions
			for (int i = 0; i < particles.size(); i++)
			{
				Particle& p = particles[i];
				p.a -= kv * (fs * p.c - fc * p.s);
			}
		}
		epot /= particles.size();

		// STEP 3: update velocities
		//    v(t + dt) = v(t) + 0.5 * dt * (a(t) + a(t + dt))
		for (int i = 0; i < particles.size(); i++)
		{
			Particle& p = particles[i];
			p.v += 0.5 * TIMESTEP * (p.aOld + p.a);
			p.aOld = p.a;
		}

		// exponential cooling
		double ekin0 = EKINSTART * pow((EKINEND / EKINSTART),  ((double)t / MDSASTEPS));

		// Andersen thermostat
		double velSTD = sqrt(ekin0);
		for (int i = 0; i < particles.size(); i++)
		{
			// update velocity with a given probability
			if (rand() % VERLETUPDATE != 0) continue;
			particles[i].v.x = randGauss(velSTD);
			particles[i].v.y = randGauss(velSTD);
		}
		
		// output particle positions periodically to .pos file (stdout)
		if (t % (MDSASTEPS / OUTPUTNUM) == 0)
		{
			std::cerr << "Step " << (t / (MDSASTEPS / OUTPUTNUM)) << " of " << OUTPUTNUM << "\n";
			
			// write header
			std::cout << "#[data]	Steps	NumParticles	Chi	Ekin0	Epot\n";
			std::cout << t << "\t";
			std::cout << particles.size() << "\t";
			std::cout << CHI << "\t";
			std::cout << ekin0 << "\t";
			std::cout << epot << "\t";
			std::cout << "#[done]\n";

			// write data
			std::cout << "dimension 2\n";
			std::cout << "box " << box.x << " " << box.y << "\n";
			std::cout << "shape \"sphere 1.0 ff0000\"\n";
			for (int i = 0; i < particles.size(); i++)
			{
				const Vector2d& pos = particles[i].x;
				std::cout << remainder(pos.x, box.x) << " ";
				std::cout << remainder(pos.y, box.y) << "\n";
			}
			std::cout << "eof\n";
			std::flush(std::cout);
		}
	}
}

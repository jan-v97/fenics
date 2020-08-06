
// Based on https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined __CYGWIN__
    #ifdef __GNUC__
        #define DLL_EXPORT __attribute__ ((dllexport))
    #else
        #define DLL_EXPORT __declspec(dllexport)
    #endif
#else
    #define DLL_EXPORT __attribute__ ((visibility ("default")))
#endif

#include <dolfin/function/Expression.h>
#include <dolfin/math/basic.h>
#include <Eigen/Dense>


// cmath functions
using std::cos;
using std::sin;
using std::tan;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cosh;
using std::sinh;
using std::tanh;
using std::exp;
using std::frexp;
using std::ldexp;
using std::log;
using std::log10;
using std::modf;
using std::pow;
using std::sqrt;
using std::ceil;
using std::fabs;
using std::floor;
using std::fmod;
using std::max;
using std::min;

const double pi = DOLFIN_PI;


namespace dolfin
{
  class dolfin_expression_871d1cfbc52d8c773eeff5b34c617b7e : public Expression
  {
     public:
       double delta_t;
double delta_r;
double L1;
double Lt;
double Lr;
double pt;
double pr;
double tlb;


       dolfin_expression_871d1cfbc52d8c773eeff5b34c617b7e()
       {
            _value_shape.push_back(2);
       }

       void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
       {
          values[0] = tlb*((y)/(L1-0.75))*((y)/(L1-0.75))+(delta_t-tlb)*((y)/(L1-0.75));
          values[1] = ((y)/(L1-0.75))*(Lt-0.75);

       }

       void set_property(std::string name, double _value) override
       {
          if (name == "delta_t") { delta_t = _value; return; }          if (name == "delta_r") { delta_r = _value; return; }          if (name == "L1") { L1 = _value; return; }          if (name == "Lt") { Lt = _value; return; }          if (name == "Lr") { Lr = _value; return; }          if (name == "pt") { pt = _value; return; }          if (name == "pr") { pr = _value; return; }          if (name == "tlb") { tlb = _value; return; }
       throw std::runtime_error("No such property");
       }

       double get_property(std::string name) const override
       {
          if (name == "delta_t") return delta_t;          if (name == "delta_r") return delta_r;          if (name == "L1") return L1;          if (name == "Lt") return Lt;          if (name == "Lr") return Lr;          if (name == "pt") return pt;          if (name == "pr") return pr;          if (name == "tlb") return tlb;
       throw std::runtime_error("No such property");
       return 0.0;
       }

       void set_generic_function(std::string name, std::shared_ptr<dolfin::GenericFunction> _value) override
       {

       throw std::runtime_error("No such property");
       }

       std::shared_ptr<dolfin::GenericFunction> get_generic_function(std::string name) const override
       {

       throw std::runtime_error("No such property");
       }

  };
}

extern "C" DLL_EXPORT dolfin::Expression * create_dolfin_expression_871d1cfbc52d8c773eeff5b34c617b7e()
{
  return new dolfin::dolfin_expression_871d1cfbc52d8c773eeff5b34c617b7e;
}


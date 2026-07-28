import Nav from "@/components/Nav";
import Hero from "@/components/Hero";
import Problem from "@/components/Problem";
import HowItWorks from "@/components/HowItWorks";
import Benchmarks from "@/components/Benchmarks";
import UsageShowcase from "@/components/UsageShowcase";
import InteractiveDemo from "@/components/InteractiveDemo";
import Quickstart from "@/components/Quickstart";
import Footer from "@/components/Footer";
import SectionDivider from "@/components/SectionDivider";

export default function Home() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <div className="mx-auto max-w-content px-6"><SectionDivider /></div>
        <Problem />
        <div className="mx-auto max-w-content px-6"><SectionDivider /></div>
        <HowItWorks />
        <div className="mx-auto max-w-content px-6"><SectionDivider /></div>
        <Benchmarks />
        <div className="mx-auto max-w-content px-6"><SectionDivider /></div>
        <UsageShowcase />
        <div className="mx-auto max-w-content px-6"><SectionDivider /></div>
        <InteractiveDemo />
        <div className="mx-auto max-w-content px-6"><SectionDivider /></div>
        <Quickstart />
      </main>
      <Footer />
    </>
  );
}

/* ═══════════════════════════════════════════════════════════
   RainLoom Landing — Interactions & Scroll Animations
   ═══════════════════════════════════════════════════════════ */

// Nav scroll
const nav = document.getElementById('nav');
window.addEventListener('scroll', () => {
  nav.classList.toggle('scrolled', window.scrollY > 40);
}, { passive: true });

// Mobile nav
const navToggle = document.getElementById('navToggle');
const navLinks = document.getElementById('navLinks');
if (navToggle) {
  navToggle.addEventListener('click', () => {
    navLinks.classList.toggle('open');
  });
}

// Smooth scroll
document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener('click', e => {
    const target = document.querySelector(link.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      navLinks.classList.remove('open');
    }
  });
});

// Scroll reveal
const initReveal = () => {
  const selectors = [
    '.info-item', '.layer', '.feature-card', '.metric-box',
    '.split-text', '.split-image', '.section-badge', '.section h2',
    '.section-sub', '.tech-strip', '.stats-strip .stat',
    '.section-cta h2', '.cta-actions',
  ];
  selectors.forEach(sel => {
    document.querySelectorAll(sel).forEach(el => {
      if (!el.classList.contains('reveal')) el.classList.add('reveal');
    });
  });
};

const checkReveal = () => {
  const vh = window.innerHeight;
  document.querySelectorAll('.reveal').forEach(el => {
    if (el.getBoundingClientRect().top < vh - 50 && !el.classList.contains('visible')) {
      // Stagger siblings
      const parent = el.parentElement;
      const siblings = parent ? [...parent.querySelectorAll('.reveal:not(.visible)')] : [];
      const idx = siblings.indexOf(el);
      setTimeout(() => el.classList.add('visible'), Math.max(0, idx) * 60);
    }
  });
};

initReveal();
window.addEventListener('scroll', checkReveal, { passive: true });
setTimeout(checkReveal, 80);

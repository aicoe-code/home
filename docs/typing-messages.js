// Typing animation messages configuration
// Define different message sets for different elements

const typingMessages = {
    // Main hero heading messages - rotates through these
    heroHeading: [
        "AI Center of Excellence",
        "Transformational AI",
        "Enterprise AI Solutions",
        "Strategic AI Implementation",
        "AI-Powered Transformation",
        "Intelligent Automation",
        "Next-Gen AI Architecture",
        "Production-Ready AI",
        "Enterprise Intelligence",
        "AI Innovation Lab",
        "Scalable AI Systems",
        "AI Strategy & Execution",
        "Modern AI Infrastructure",
        "AI Excellence Framework"
    ],

    // Subtitle messages - rotates through these
    heroSubtitle: [
        "Transformational AI implementation. From strategic assessment to production-ready solutions.",
        "Strategic consulting. Production-ready architecture. Enterprise transformation.",
        "Building the future of AI-powered organizations.",
        "Accelerating innovation through intelligent systems and scalable architectures.",
        "Driving efficiency with enterprise-grade automation and AI solutions.",
        "Modern AI infrastructure. Cloud-native deployment. Production excellence.",
        "From proof-of-concept to production. Delivering AI at enterprise scale.",
        "Strategic roadmaps. Technical excellence. Measurable business outcomes.",
        "Empowering teams with cutting-edge AI capabilities and innovation frameworks.",
        "Architecting intelligent systems. Scaling AI initiatives. Transforming operations.",
        "End-to-end AI delivery. Assessment to deployment. Strategy to execution.",
        "Building resilient AI platforms. Modern architectures. Sustainable innovation.",
        "Enterprise AI expertise. Technical depth. Strategic vision.",
        "Transforming businesses through intelligent automation and advanced AI systems."
    ],

    // Single static message (no rotation)
    singleMessage: [
        "AI Center of Excellence"
    ],

    // Configuration options
    config: {
        typingSpeed: 50,        // milliseconds per character
        deletingSpeed: 30,      // milliseconds per character when deleting
        delayBetweenMessages: 2000,  // pause before deleting and showing next message
        initialDelay: 500,      // delay before typing starts
        cursorBlinkSpeed: 530   // cursor blink animation speed
    }
};

// Export for use in main page
if (typeof module !== 'undefined' && module.exports) {
    module.exports = typingMessages;
}

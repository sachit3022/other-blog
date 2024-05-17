+++
title = 'Notes on building a product in public'
date = 2024-05-16T11:55:02-04:00
draft = false
+++


### Coming up with ideas for the types of products I want to build

In recent times, there has been a lot of excitement about GenAI products. This is also one of my strong points, as I am well-versed in the literature. GenAI is very broad, and we would need to pick an area to focus on. 

Drawing from my learnings from books and working in multiple startups, B2C startups are very challenging and suffer from cold start problems, requiring VC funding to push forward. However, as I am not a proven entrepreneur, I lack the luxury of VC money, and I believe it would be a distraction at the moment. Therefore, we narrow our focus to B2B.

One of my inspirations is BAIR at UC Berkeley; they excel at translating their research into products. Now, we need to focus on making the product sellable. I cannot build general-purpose models as training them is expensive, and looking at some of the successes of OpenAI and other companies, they succeed in building genera purpose models. However, the ability to customize models is crucial and will not be the focus area of big companies.

Now, let's look at the recent funding received by AI companies and see what they are missing. Thanks to [Chief AI Office](https://www.chiefaioffice.xyz/c/database) for releasing funding reports of AI companies. From my initial review, many B2B companies have not targeted the customization segment. Some examples include Glean, which improves search on company documents, and Descript, which focuses on editing marketing collateral.

Having worked with marketing companies at Haptik, I have noted that full automation will not be the primary focus of large companies initially. They focus on tools to make their lives easier, instantly applicable to their existing problems. For instance, many apparel companies, like [Nike](https://www.nike.com/t/pegasus-40-mens-road-running-shoes-zD8H1c/DV3853-002), Amazon, and Shopify stores, sell apparel with many variants, yet they often have marketing collateral for only one variant due to the expense and time-consuming nature of creating such collateral. However, they we have images for all the products but not on models. If we can build a product that solves this problem, we can expand further into the virtual try-on space, enabling customers to visualize the product on themselves. 

There is immense potential for this product, which starts small. However, the risk lies in the technology not being there yet. Despite the surge in image generation due to the rise of diffusion models, which generate high-fidelity images, the challenge remains: can we combine ideas from multiple papers and build a product that works? Another risk is whether companies are not editing marketing collateral manually because buyers do not care about how it looks on models or can easily infer from two images. If buyers do not care, companies will not invest in the product.

If big players like Nike do not prioritize it, will smaller ones who sell products via Shopify care? Which market should we target? Before finding answers to these questions, let me build the product with a deadline within two weeks with a basic version and reach out to large corporates like Nike, Adidas, Puma, New Balance, and Allbirds, or perhaps apply for Y Combinator, demonstrating the technology and learning market creation from YC.

### Upskill and cover the bases

The best way to learn is to have skin in the game. Now that we have narrowed down the idea, it's time to get into heads-down execution and build the product. I will build on existing knowledge of diffusion models, understand them, and outline them in a blog on [diffusion models](https://sachit3022.github.io/other-blog/posts/diffusion/). Later, I will expand on conditional generation, try out toy problems, and demonstrate that the idea is feasible with current technology before expanding to the real use case of apparel. Today, we work with diffusion; tomorrow, a new idea may arise. This passion for developing products facilitates learning and knowledge sticks for long term, as I am invested in the learning journey.

Several research papers that I will implement have accompanying code, and their models are public - [DDPM](https://arxiv.org/pdf/2006.11239), [Stable Diffusion](https://stability.ai/news/stable-diffusion-v2-release), [DreamBooth](https://dreambooth.github.io/), [SDEdit](https://arxiv.org/pdf/2108.01073). The challenge lies in combining them. I will note the implementation and insights on blogs on the respective topics.

### Cherry-pick and create a demo

To convince anyone—clients, customers, or YC—we need to demonstrate the success of the product.


Display Image credit:  <a href="https://iconscout.com/icons/website-builder" class="text-underline font-size-sm" target="_blank">website builder</a> by <a href="https://iconscout.com/contributors/WHCompare" class="text-underline font-size-sm">Alexiuz As</a> on <a href="https://iconscout.com" class="text-underline font-size-sm">IconScout</a>

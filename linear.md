# seeing scaling laws in action

We need to train a set of dense models and then a set of MoEs and compare their token's per second to figure out if training a MoE is worth it.

First, I started training llama2_271M on my vast instance for 15000 steps, should be enough to saturate the loss curve. after about 350 steps, this is the loss curve (saturated around 25 steps ):

![image.png](https://uploads.linear.app/95ec02cf-44cd-4674-b400-07199068c174/28496bc4-e778-4433-86d2-9553f9b9f0c4/e7f0babc-08c2-48d9-8608-66624aa27bdf)

looks like the scaling law is kicking in, next step is to scale up the model. 

this is the same model but with d_model set to 2048, it seemed to saturate even faster (20 steps):

![image.png](https://uploads.linear.app/95ec02cf-44cd-4674-b400-07199068c174/af2301f8-257c-4b1c-9439-14c13317072e/50417d4d-da29-4fbb-a1d2-181f85eb60b6)

applying a log scale to both axes (as in OpenAI's paper) reveals that I made a mistake with the smaller model, and need to rerun it. Here's llama2_2x271M for 500 steps:

![image.png](https://uploads.linear.app/95ec02cf-44cd-4674-b400-07199068c174/2ed7b640-089f-4bea-809a-c64754eab7b4/8c99933c-95c5-4686-b0fe-bce4ba35ea50)

we see that the loss bottoms out at around (265, 5.65), then it goes up a bit and stagnates. Here's the same scale applied to the smaller 271M model:

![image.png](https://uploads.linear.app/95ec02cf-44cd-4674-b400-07199068c174/3024cd91-a0ff-4dbb-aa86-7c159ee5743b/0ec4e0e2-9717-407d-bf50-47c76f8cdfda)

 This means I stopped the 271M model early, since we don't see loss bottom out. 

superimposing the graphs on a log scale gives a weird contradictory result:

![superimposed graphs.png](https://uploads.linear.app/95ec02cf-44cd-4674-b400-07199068c174/3094379e-45b0-4804-b404-23f5ab74d2c0/d3213332-dd95-40a9-8f3b-ee196fccecd5)

the smaller model reaches a better performance, and doesn't saturate. the only difference here is d_model. 

update Wednesday 6pm: the 135M and 271M have effectively saturated wrt to the openAI limit within a reasonable margin. the 1B variant of the same model is close to saturation as well. here's how that looks like on wandb:

![image.png](https://uploads.linear.app/95ec02cf-44cd-4674-b400-07199068c174/21af3b4f-525e-4955-96ff-364eb006e20e/d9f9e656-8a55-4211-ae8f-4528e9b6593d)

Here is claude showing the loss curve wrt the scaling law openAI came up with:

![image.png](https://uploads.linear.app/95ec02cf-44cd-4674-b400-07199068c174/41661f53-87cb-49c5-bc5f-b7231faa78bb/f00d2df9-ae33-4a4c-a201-6cd218c1f988)

![image.png](https://uploads.linear.app/95ec02cf-44cd-4674-b400-07199068c174/e4640c71-86d9-4906-bfcc-a2710fb43e7c/32a19f88-36da-4fcc-9f6a-b61e6bd9d727)

![image.png](https://uploads.linear.app/95ec02cf-44cd-4674-b400-07199068c174/d65d5b02-842c-4455-950f-a2617c79fb92/5e31e2ab-36b2-44f5-acd0-f451585233b3)

I went ahead and spun up 2 MoE of total param size 1B, with activation ratios \~39% and \~10%. here's how all of them compare wrt to openAI's limit:

![image.png](https://uploads.linear.app/95ec02cf-44cd-4674-b400-07199068c174/b102c36a-7356-40cd-af99-d2b4df6d4683/55149962-581a-4e6b-92f2-149ebd09f68d)

we see that the bigger model (271M) bottoms out later than the smaller one (135). we also see that the moe counterparts achieve a lower loss than the 1B models with the same amout of compute. the question is, will they bottom out before or after the dense 1B? 

in terms of FLOPs, we see that indeed the sparse 1B model is the cheapest, 2nd place is the less sparse moe, and 3rd is the dense 1B:

![image.png](https://uploads.linear.app/95ec02cf-44cd-4674-b400-07199068c174/490d884d-6440-483e-8dbe-4d35973d3334/ed4b2837-90aa-4285-85fe-155fca0cc4d2)

immediate next step is to analyse this data to get clear flop cost ratios wrt to activation ratio and total params.

I'd say we have successfully observed that scaling limit, but more experiments are needed to get a better feeling for the training dynamics of MoE's wrt their Dense counterparts. namely:

if we make a MoE with 1B active params (with say 2.5B total and 10B total), will it outperform a dense 1B? what about a 10B dense? How much cheaper does it run in terms of FlOPs? what if we go even more sparse? according to theory we should get more gains, but I coulnd't do it at the 1B scale because the always on params are \~100M minimum at 1B, meaning my current lowest activation ratio is \~10%. 

Thursday morning update: the dense 1B is still not done, I ran a 60M and a 1M dense model, and the sparse ones mostly saturated. Here's the plot of loss vs PF days:

![image.png](https://uploads.linear.app/95ec02cf-44cd-4674-b400-07199068c174/207cdfc2-f591-4363-a353-7a248b8afb1d/9d94c7d4-816a-4fa9-8c14-1e7ff4dfcb9d)

conclusions: under equal total params, MoEs train faster and better, but it's still not clear whether they bottom out later as well. comparing MoE 1B sparse (\~100M active params) to 135M dense, it seems that MoE's might also bottom out later. we also still don't know when the dense 1B bottoms out. Here is the same plot but using OLMo's FLOP count:

![image.png](https://uploads.linear.app/95ec02cf-44cd-4674-b400-07199068c174/3d3e26ee-2f1e-4139-8710-1be6e8270bb8/979c00e2-91b8-4c5f-96ef-8a1cc8efd9b2)

this paper: [https://arxiv.org/abs/2502.05172](https://arxiv.org/abs/2502.05172) attempts to define a joint scaling law for loss wrt number of active parameters and dataset size. In the following plot, it shows that there exists a compute regime for which training an MoE gives better results than training an equally sized dense model (1.1B params). HOWEVER, you can see on the right end of the plot that overtraining the dense version resulted in lower loss anyway. 

![image.png](https://uploads.linear.app/95ec02cf-44cd-4674-b400-07199068c174/ba715885-fe20-4eeb-afd0-18f47f0063bb/c94f242a-3c65-48bf-8e01-bd90aaaceade)

Tokens per second discussion

TPS for the sparse MoE is 93k, while a similarly sized dense model is 180k. 

TPs for olmo MoE is 50k, while a similarly sized dense model is 90k. 

Why? the hypothesis is memory bandwidth of 5090. 

new data: on H100, with the same microbatch size, we get 142k on the sparse model, but 415k on dense! thats a 2.92x increase vs a 1.93x on the rtx5090.

tried to get a 7B running on nebius, failed. even put microbatch all the way down to 512 tokens and edited max sequence length

friday after morning sync:

the best thing to do rn is: plot the log loss over PF days for dense, and then have the same plot for the MoE models. 

big question: is what we've seen with MoEs so far worth a super scaled experiment?

curiosity: why is the knee so much more prominent in openAI?

my immediate task is plotting dense and moe experiments nicely. 

my secondary task is running more moe training runs. 

**FRIDAY FINAL EVALUATION:**

this week, I trained 9 dense transformers and 7 MoE's as much as I could given 15B tokens of the c4 dataset, 5 RTX5090s and 5 H100 gpus (always 1 gpu per run). My first goal was validating that there exists a scaling law for log training loss vs compute in flops for increasing sizes of dense transformers. This is log loss vs compute in flops, for all dense models:

![image.png](https://uploads.linear.app/95ec02cf-44cd-4674-b400-07199068c174/0a631470-516d-48b9-beb3-89a89de53202/43df5c1c-3d70-4789-ba32-b75d1e321140)

Conclusion: you can clearly see some kind of scaling law emerging, but it's at odds with OpenAI's. One hypothesis is that this is caused by the difference in datasets.

Also, we operate in a much larger compute regime (<=10^4 PF-days), while OpenAI operates at <=10^1.

![image.png](https://uploads.linear.app/95ec02cf-44cd-4674-b400-07199068c174/1a4fc717-d80d-4434-9617-b96df44cbf1b/f21baa65-7c78-40c6-894b-4c2c4497a619)

So, how do the MoE's perform? when comparing total parameters, for a fixed amount of compute, the MoE consistently reach a lower loss than an equivalently sized dense model. Also, the amount by which the loss decreased increases with total parameters (horizontal gap btw blue is thin, green is wider, red is widest) each colour family in the following plot represents a different total parameters (denser means darker):

![image.png](https://uploads.linear.app/95ec02cf-44cd-4674-b400-07199068c174/ddc01e9d-57ff-4745-a248-bb92f1c1275f/a6763fd5-74cb-4ca0-99ce-b8d446af04fb)

In terms of active parameters, a given MoE performs very similar to or better than, the equivalent dense model, as long as active parameter count >500M: Here each colour family represents a similar amount of active parameters:  

![image.png](https://uploads.linear.app/95ec02cf-44cd-4674-b400-07199068c174/2fd0bad2-ebef-471e-8df5-4c9edb604e0a/b86ef242-0ea2-4037-a380-17f5bf292511)

Is the scaling law better for MoEs? perhaps, but we'd need to train larger models to start to see the gains:

![image.png](https://uploads.linear.app/95ec02cf-44cd-4674-b400-07199068c174/72b71a5e-a1bf-4b61-a443-cb89518cdb00/08497ebc-68a8-4296-a3b2-07fb3c253873)

## Metadata
- URL: [https://linear.app/relace/issue/REL-374/seeing-scaling-laws-in-action](https://linear.app/relace/issue/REL-374/seeing-scaling-laws-in-action)
- Identifier: REL-374
- Status: Backlog
- Priority: No priority
- Assignee: Unassigned
- Project: [Pre & Mid-training](https://linear.app/relace/project/pre-and-mid-training-619607b99ea5). 
- Project milestone: Create and test MoE model in training framework
- Created: 2026-01-14T03:14:00.505Z
- Updated: 2026-01-17T02:04:55.137Z

## Sub-issues

- [REL-403 Demystifying scaling laws.](https://linear.app/relace/issue/REL-403/demystifying-scaling-laws)